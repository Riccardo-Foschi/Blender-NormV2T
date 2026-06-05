# refs: bake ops depend on active+selected context; bit depth via ImageFormatSettings.color_depth

bl_info = {
    "name": "Normal-v2t Baker",
    "author": "Riccardo Foschi + Perplexity/Gemini 3.1 Pro",
    "version": (2, 28),
    "blender": (4, 0, 0),
    "location": "View3D > Sidebar > NORM-v2t",
    "description": "Project View space normals into a mesh from specific camera positions, the normals and the cameras must have same name",
    "category": "Object",
}

import bpy
import numpy as np
import os
import math
import uuid
import time
import json
import base64
from array import array
from bpy.types import Operator, Panel, PropertyGroup
from bpy.props import (
    StringProperty, EnumProperty, IntProperty, BoolProperty,
    FloatProperty, PointerProperty
)

# ------------------------------------------------------------------------
# DATA CLASSES
# ------------------------------------------------------------------------

def poll_is_mesh(self, obj):
    return (obj is not None) and isinstance(obj, bpy.types.Object) and (obj.type == 'MESH')

class BakeSettings(PropertyGroup):
    show_advanced: BoolProperty(
        name="Advanced Parameters",
        default=False,
        description="Show advanced projection and culling parameters"
    )

    target_mesh: PointerProperty(
        name="Target Mesh",
        type=bpy.types.Object,
        poll=poll_is_mesh,
        description="Object to bake onto"
    )

    input_path: StringProperty(name="Input Folder", default="//set normal map to project here", subtype='DIR_PATH')
    output_path: StringProperty(name="Output Folder", default="//set output folder here", subtype='DIR_PATH')

    save_debug_images: BoolProperty(
        name="Save Debug Images",
        default=True,
        description="Save intermediate maps used for slope correction"
    )

    use_corrected_map_as_new_input: BoolProperty(
        name="Use Corrected Maps as New Input",
        default=True,
        description="After correcting the slope, set the output folder as the new input folder so subsequent operations use the corrected maps"
    )

    res_x: IntProperty(name="Res X", default=2048, min=64, max=16384)
    res_y: IntProperty(name="Res Y", default=2048, min=64, max=16384)

    angle_limit: FloatProperty(
        name="Angle Limit",
        description="Exclude faces angled more than this degrees away from camera",
        default=math.radians(89.0),
        min=0.0,
        max=math.pi,
        subtype='ANGLE',
        unit='ROTATION'
    )

    occlusion_samples: IntProperty(
        name="Occlusion Samples",
        description=(
            "Ray samples per face for occlusion culling. "
            "1 = center ray only (fast). "
            "2+ = center + all face vertices, majority vote (robust against partial occlusion)"
        ),
        default=1,
        min=1,
        max=8
    )

    occlusion_tolerance: FloatProperty(
        name="Occlusion Tolerance",
        description="Distance tolerance for ray hit. Lower = more rigid culling. (0.0001 is very rigid)",
        default=0.0001,
        min=0.0,
        max=1.0,
        precision=5
    )

    occlusion_expand_steps: IntProperty(
        name="Occlusion Mask Bleed",
        description="Expand the occlusion mask by N connected face rings to prevent spill-over at silhouette edges",
        default= 2,
        min=0,
        max=5
    )

    file_format: EnumProperty(
        name="Format",
        items=[
            ('PNG', "PNG", ""),
            ('JPEG', "JPEG", ""),
            ('TARGA', "Targa", ""),
            ('OPEN_EXR', "OpenEXR", ""),
            ('TIFF', "TIFF", "")
        ],
        default='TIFF'
    )

    color_depth: EnumProperty(
        name="Bit Depth",
        items=[('8', "8 Bit", ""), ('16', "16 Bit", "")],
        default='8'
    )

    blend_input_to_object_normals: BoolProperty(
        name="Blend Input to Object Normals",
        default=False,
        description="Render camera/view-space normals of the target mesh, blend with input normal maps, and use as input"
    )

    create_composite: BoolProperty(
        name="Generate Composite Map",
        default=True,
        description="Merges all maps onto a 'Flat Normal' (Purple) background"
    )

    apply_to_mesh: BoolProperty(
        name="Apply Baked Normal to Mesh",
        default=True,
        description="Create and assign a material with the final normal map"
    )

    update_original_material: BoolProperty(
        name="Update Original Material",
        default=True,
        description="If enabled, restores the original material and plugs the baked normal map into it"
    )

    debug_image_index: IntProperty(
        name="Debug Image Index",
        description="1-based index into the images found in the Input folder",
        default=1,
        min=1
    )

# ------------------------------------------------------------------------
# UTILS
# ------------------------------------------------------------------------

# ------------------------------------------------------------------------
# BLEND PIPELINE FUNCTIONS
# ------------------------------------------------------------------------

def _create_camspace_normal_material(mat_name="BNBP_CamSpace_Normal_Mat"):
    mat = bpy.data.materials.get(mat_name)
    if mat:
        try: bpy.data.materials.remove(mat)
        except: pass

    mat = bpy.data.materials.new(name=mat_name)
    mat.use_nodes = True
    nt = mat.node_tree
    nodes = nt.nodes
    links = nt.links
    nodes.clear()

    node_geo = nodes.new(type="ShaderNodeNewGeometry")
    node_geo.location = (-1000, 0)

    node_transform = nodes.new(type="ShaderNodeVectorTransform")
    node_transform.vector_type = 'VECTOR'
    node_transform.convert_from = 'WORLD'
    node_transform.convert_to = 'CAMERA'
    node_transform.location = (-800, 0)

    node_sep = nodes.new(type="ShaderNodeSeparateXYZ")
    node_sep.location = (-600, 0)

    node_math = nodes.new(type="ShaderNodeMath")
    node_math.operation = 'MULTIPLY'
    node_math.inputs[1].default_value = -1.0
    node_math.location = (-400, -200)

    node_comb = nodes.new(type="ShaderNodeCombineXYZ")
    node_comb.location = (-200, 0)

    node_vm = nodes.new(type="ShaderNodeVectorMath")
    node_vm.operation = 'MULTIPLY_ADD'
    node_vm.inputs[1].default_value = (0.5, 0.5, 0.5)
    node_vm.inputs[2].default_value = (0.5, 0.5, 1.0)
    node_vm.location = (0, 0)

    node_em = nodes.new(type="ShaderNodeEmission")
    node_em.inputs['Strength'].default_value = 1.0
    node_em.location = (200, 0)

    node_out = nodes.new(type="ShaderNodeOutputMaterial")
    node_out.location = (400, 0)

    links.new(node_geo.outputs['Normal'], node_transform.inputs['Vector'])
    links.new(node_transform.outputs['Vector'], node_sep.inputs['Vector'])
    links.new(node_sep.outputs['X'], node_comb.inputs['X'])
    links.new(node_sep.outputs['Y'], node_comb.inputs['Y'])
    links.new(node_sep.outputs['Z'], node_math.inputs[0])
    links.new(node_math.outputs['Value'], node_comb.inputs['Z'])
    links.new(node_comb.outputs['Vector'], node_vm.inputs[0])
    links.new(node_vm.outputs['Vector'], node_em.inputs['Color'])
    links.new(node_em.outputs['Emission'], node_out.inputs['Surface'])

    return mat

def _blend_normals_overlay_renorm(rendered_path, input_path, output_path, file_format):
    img_rend = bpy.data.images.load(rendered_path, check_existing=False)
    img_rend.colorspace_settings.name = 'Non-Color'

    img_in = bpy.data.images.load(input_path, check_existing=False)
    img_in.colorspace_settings.name = 'Non-Color'

    w, h = img_rend.size
    if img_in.size[0] != w or img_in.size[1] != h:
        img_in.scale(w, h)

    p_r = np.empty((w * h * 4,), dtype=np.float32)
    p_i = np.empty((w * h * 4,), dtype=np.float32)
    img_rend.pixels.foreach_get(p_r)
    img_in.pixels.foreach_get(p_i)

    arr_r = p_r.reshape((-1, 4))
    arr_i = p_i.reshape((-1, 4))

    bg_rgb = np.array([128.0/255.0, 128.0/255.0, 255.0/255.0], dtype=np.float32)

    a_r = arr_r[:, 3:4]
    a_i = arr_i[:, 3:4]

    rgb_r = arr_r[:, :3] * a_r + bg_rgb * (1.0 - a_r)
    rgb_i = arr_i[:, :3] * a_i + bg_rgb * (1.0 - a_i)

    mask = rgb_r < 0.5
    rgb = np.where(mask, 2.0 * rgb_r * rgb_i, 1.0 - 2.0 * (1.0 - rgb_r) * (1.0 - rgb_i))

    vec = rgb * 2.0 - 1.0
    n = np.linalg.norm(vec, axis=1, keepdims=True)
    n[n == 0] = 1.0
    vec = vec / n
    rgb_norm = vec * 0.5 + 0.5

    out_alpha = np.ones((rgb_norm.shape[0], 1), dtype=np.float32)
    out = np.concatenate((rgb_norm, out_alpha), axis=1).astype(np.float32)

    img_out = bpy.data.images.new(name=_uid("BNBP_blended"), width=w, height=h, alpha=True)
    img_out.colorspace_settings.name = 'Non-Color'
    img_out.pixels.foreach_set(out.ravel())

    img_out.filepath_raw = output_path
    img_out.file_format = file_format
    img_out.save()

    try: bpy.data.images.remove(img_rend)
    except: pass
    try: bpy.data.images.remove(img_in)
    except: pass
    try: bpy.data.images.remove(img_out)
    except: pass

def _prepare_blended_input_normals(context, mesh_obj, valid_pairs, out_dir, props):
    scene = context.scene

    debug_dir = os.path.join(out_dir, "mesh_CamSpace_Normals_Debug")
    blend_dir = os.path.join(out_dir, "blended_input_to_object_normals")
    os.makedirs(debug_dir, exist_ok=True)
    os.makedirs(blend_dir, exist_ok=True)

    old_camera = scene.camera
    old_engine = scene.render.engine
    old_filepath = scene.render.filepath
    old_format = scene.render.image_settings.file_format
    
    # SALVATAGGIO VECCHIO MEDIA TYPE
    old_media_type = getattr(scene.render.image_settings, "media_type", 'IMAGE')
    
    old_color_mode = scene.render.image_settings.color_mode
    old_transparent = scene.render.film_transparent
    old_view_transform = scene.view_settings.view_transform
    old_res_x = scene.render.resolution_x
    old_res_y = scene.render.resolution_y
    old_res_pct = scene.render.resolution_percentage

    mesh_state = backup_mesh_material_state(mesh_obj)
    cam_mat = None

    try:
        try:
            scene.render.engine = 'BLENDER_EEVEE_NEXT'
        except:
            scene.render.engine = 'BLENDER_EEVEE'
            
        # --- FORCE IMAGE TYPE TO BYPASS VIDEO FORMATS ---
        try:
            scene.render.image_settings.media_type = 'IMAGE'
        except AttributeError:
            pass
            
        scene.render.image_settings.file_format = 'PNG'
        try:
            scene.render.image_settings.color_mode = 'RGBA'
        except: pass
        
        scene.render.film_transparent = True
        scene.view_settings.view_transform = 'Raw'
        scene.render.resolution_percentage = 100

        cam_mat = _create_camspace_normal_material()

        if len(mesh_obj.material_slots) == 0:
            mesh_obj.data.materials.append(cam_mat)
        else:
            for slot in mesh_obj.material_slots:
                slot.material = cam_mat

        new_pairs = []
        ext = '.png'

        for cam_obj, input_img_path in valid_pairs:

            rx, ry = None, None
            try:
                rx, ry = get_image_resolution_from_disk(input_img_path)
            except:
                pass

            if rx and ry:
                scene.render.resolution_x = int(rx)
                scene.render.resolution_y = int(ry)

            scene.camera = cam_obj

            context.view_layer.update()
            bpy.ops.wm.redraw_timer(type='DRAW_WIN_SWAP', iterations=1)

            render_path = os.path.join(debug_dir, f"{cam_obj.name}_rendered{ext}")
            scene.render.filepath = render_path

            bpy.ops.render.render(write_still=True)

            wait_for_file_ready(render_path, timeout=10.0)

            blended_path = os.path.join(blend_dir, f"{cam_obj.name}{ext}")
            _blend_normals_overlay_renorm(render_path, input_img_path, blended_path, 'PNG')

            wait_for_file_ready(blended_path, timeout=10.0)

            new_pairs.append((cam_obj, blended_path))

        return new_pairs

    finally:
        try: restore_mesh_material_state(mesh_obj, mesh_state)
        except: pass

        try:
            if cam_mat and cam_mat.name in bpy.data.materials:
                bpy.data.materials.remove(cam_mat)
        except: pass

        scene.camera = old_camera
        scene.render.engine = old_engine
        scene.render.filepath = old_filepath
        
        # RIPRISTINO VECCHIO MEDIA TYPE
        try:
            scene.render.image_settings.media_type = old_media_type
        except AttributeError:
            pass
            
        scene.render.image_settings.file_format = old_format
        scene.render.image_settings.color_mode = old_color_mode
        scene.render.film_transparent = old_transparent
        scene.view_settings.view_transform = old_view_transform
        scene.render.resolution_x = old_res_x
        scene.render.resolution_y = old_res_y
        scene.render.resolution_percentage = old_res_pct

        context.view_layer.update()

# ------------------------------------------------------------------------
# UTILS
# ------------------------------------------------------------------------

def _uid(prefix="TEMP"):
    return f"{prefix}_{uuid.uuid4().hex[:8]}"

def safe_remove_datablock(collection, datablock):
    if not datablock:
        return
    try:
        if datablock.users == 0:
            collection.remove(datablock)
            return
    except:
        pass
    try:
        collection.remove(datablock)
    except:
        pass

def ensure_active_selected(context, obj):
    vl = context.view_layer

    try:
        if vl.objects.active is None:
            try:
                bpy.ops.object.select_all(action='DESELECT')
            except:
                pass
            try:
                obj.select_set(True)
            except:
                pass
            try:
                vl.objects.active = obj
            except:
                pass
    except:
        pass

    try:
        if vl.objects.active is not None:
            bpy.ops.object.mode_set(mode='OBJECT')
    except:
        pass

    try:
        bpy.ops.object.select_all(action='DESELECT')
    except:
        pass
    try:
        obj.select_set(True)
    except:
        pass
    try:
        vl.objects.active = obj
    except:
        pass

def backup_selection_state(context):
    vl = context.view_layer
    sel_names = [o.name for o in context.selected_objects]
    active_name = vl.objects.active.name if vl.objects.active else None
    mode = vl.objects.active.mode if vl.objects.active else 'OBJECT'
    return {"sel_names": sel_names, "active_name": active_name, "mode": mode}

def restore_selection_state(context, st):
    try:
        bpy.ops.object.select_all(action='DESELECT')
    except:
        pass
    for name in st["sel_names"]:
        o = bpy.data.objects.get(name)
        if o:
            try:
                o.select_set(True)
            except:
                pass
    act = bpy.data.objects.get(st["active_name"]) if st["active_name"] else None
    try:
        context.view_layer.objects.active = act
    except:
        pass
    try:
        bpy.ops.object.mode_set(mode='OBJECT')
        if act and st["mode"] in {'OBJECT', 'EDIT', 'SCULPT', 'VERTEX_PAINT', 'WEIGHT_PAINT', 'TEXTURE_PAINT'}:
            bpy.ops.object.mode_set(mode=st["mode"])
    except:
        pass

def backup_mesh_material_state(mesh_obj):
    orig_mats = list(mesh_obj.data.materials)
    polys = mesh_obj.data.polygons
    orig_poly_mi = array('I', [0]) * len(polys)
    if len(polys) > 0:
        polys.foreach_get("material_index", orig_poly_mi)
    active_index = mesh_obj.active_material_index
    return {"orig_mats": orig_mats, "orig_poly_mi": orig_poly_mi, "active_index": active_index}

def restore_mesh_material_state(mesh_obj, st):
    mesh_obj.data.materials.clear()
    for m in st["orig_mats"]:
        mesh_obj.data.materials.append(m)
    polys = mesh_obj.data.polygons
    if len(polys) == len(st["orig_poly_mi"]) and len(polys) > 0:
        polys.foreach_set("material_index", st["orig_poly_mi"])
    mesh_obj.data.update()
    try:
        if len(mesh_obj.data.materials) > 0:
            mesh_obj.active_material_index = max(0, min(int(st.get("active_index", 0)), len(mesh_obj.data.materials) - 1))
        else:
            mesh_obj.active_material_index = 0
    except:
        pass

def debug_backup_store_on_object(mesh_obj):
    try:
        if mesh_obj.get("_BNBP_DEBUG_BACKUP", None):
            return
    except:
        pass

    st = backup_mesh_material_state(mesh_obj)
    mats = []
    for m in st["orig_mats"]:
        mats.append(m.name if m else "")

    poly_bytes = b""
    try:
        poly_bytes = st["orig_poly_mi"].tobytes()
    except:
        poly_bytes = b""

    data = {
        "mats": mats,
        "active_index": int(st.get("active_index", 0)),
        "poly_len": int(len(st["orig_poly_mi"])) if "orig_poly_mi" in st else 0,
        "poly_b64": base64.b64encode(poly_bytes).decode("ascii") if poly_bytes else ""
    }

    try:
        mesh_obj["_BNBP_DEBUG_BACKUP"] = json.dumps(data)
    except:
        pass

def debug_backup_restore_from_object(mesh_obj):
    raw = None
    try:
        raw = mesh_obj.get("_BNBP_DEBUG_BACKUP", None)
    except:
        raw = None
    if not raw:
        return False

    try:
        data = json.loads(raw)
    except:
        return False

    mats = data.get("mats", [])
    active_index = int(data.get("active_index", 0))
    poly_len = int(data.get("poly_len", 0))
    poly_b64 = data.get("poly_b64", "")

    mesh_obj.data.materials.clear()
    for nm in mats:
        if not nm:
            continue
        m = bpy.data.materials.get(nm)
        if m:
            mesh_obj.data.materials.append(m)

    try:
        if poly_b64 and poly_len == len(mesh_obj.data.polygons) and len(mesh_obj.data.polygons) > 0:
            b = base64.b64decode(poly_b64.encode("ascii"))
            arr = array('I')
            arr.frombytes(b)
            if len(arr) == len(mesh_obj.data.polygons):
                mesh_obj.data.polygons.foreach_set("material_index", arr)
                mesh_obj.data.update()
    except:
        pass

    try:
        if len(mesh_obj.data.materials) > 0:
            mesh_obj.active_material_index = max(0, min(active_index, len(mesh_obj.data.materials) - 1))
        else:
            mesh_obj.active_material_index = 0
    except:
        pass

    try:
        del mesh_obj["_BNBP_DEBUG_BACKUP"]
    except:
        pass

    return True

def set_any_3d_view_to_camera_view():
    wm = bpy.context.window_manager
    for window in wm.windows:
        screen = window.screen
        for area in screen.areas:
            if area.type != 'VIEW_3D':
                continue

            try:
                for space in area.spaces:
                    if space.type == 'VIEW_3D' and getattr(space, "region_3d", None):
                        space.region_3d.view_perspective = 'CAMERA'
            except:
                pass

            try:
                region = None
                for r in area.regions:
                    if r.type == 'WINDOW':
                        region = r
                        break
                if region:
                    override = {'window': window, 'screen': screen, 'area': area, 'region': region}
                    try:
                        bpy.ops.view3d.view_camera(override)
                        return True
                    except:
                        pass
            except:
                pass

    return True
    return False

def wait_for_file_ready(path, timeout=10.0, stable_checks=3, interval=0.12):
    t0 = time.time()
    last_size = -1
    stable = 0
    while time.time() - t0 < timeout:
        try:
            if os.path.exists(path):
                sz = os.path.getsize(path)
                if sz > 0 and sz == last_size:
                    stable += 1
                    if stable >= stable_checks:
                        return True
                else:
                    stable = 0
                    last_size = sz
        except:
            pass
        time.sleep(interval)
        try:
            bpy.ops.wm.redraw_timer(type='DRAW_WIN_SWAP', iterations=1)
        except:
            pass
    return os.path.exists(path)

def _make_temp_scene_for_saving(file_format, color_depth):
    temp_scene = bpy.data.scenes.new(_uid("TEMP_SaveScene"))
    try:
        iset = temp_scene.render.image_settings

        # IMPOSTA MEDIA TYPE PRIMA DEL FILE FORMAT
        try:
            iset.media_type = 'IMAGE'
        except AttributeError:
            pass

        try:
            iset.file_format = file_format
        except:
            pass

        try:
            iset.color_depth = color_depth
        except:
            pass

        try:
            iset.color_mode = 'RGBA'
        except:
            pass

        try:
            temp_scene.view_settings.view_transform = 'Standard'
            temp_scene.view_settings.look = 'None'
            temp_scene.view_settings.exposure = 0.0
            temp_scene.view_settings.gamma = 1.0
        except:
            pass

        return temp_scene
    except:
        try:
            bpy.data.scenes.remove(temp_scene)
        except:
            pass
        return None

def save_image_robust(img, filepath, file_format, color_depth):
    if not img:
        return False

    img.filepath_raw = filepath

    temp_scene = _make_temp_scene_for_saving(file_format, color_depth)
    try:
        if temp_scene:
            try:
                img.save_render(filepath, scene=temp_scene)
                return wait_for_file_ready(filepath)
            except:
                pass

        try:
            if hasattr(img, "file_format"):
                try:
                    img.file_format = file_format
                except:
                    pass
            img.save()
            return wait_for_file_ready(filepath)
        except:
            return False
    finally:
        if temp_scene:
            try:
                bpy.data.scenes.remove(temp_scene)
            except:
                pass

def reimport_image_from_disk(filepath):
    abs_path = bpy.path.abspath(filepath)

    to_remove = []
    for im in bpy.data.images:
        try:
            if bpy.path.abspath(im.filepath) == abs_path:
                to_remove.append(im)
        except:
            continue
    for im in to_remove:
        safe_remove_datablock(bpy.data.images, im)

    wait_for_file_ready(abs_path, timeout=10.0)

    try:
        img = bpy.data.images.load(abs_path, check_existing=False)
        try:
            img.colorspace_settings.name = 'Non-Color'
        except:
            pass
        try:
            img.reload()
        except:
            pass
        return img
    except:
        return None

def show_message_box(message="", title="Alert", icon='INFO'):
    def draw(self, context):
        for line in str(message).splitlines():
            self.layout.label(text=line)
    try:
        bpy.context.window_manager.popup_menu(draw, title=title, icon=icon)
    except:
        pass

def get_folder_images_sorted(in_dir):
    valid_extensions = ('.jpg', '.jpeg', '.png', '.tga', '.exr', '.tif', '.tiff')
    try:
        files = [f for f in os.listdir(in_dir) if f.lower().endswith(valid_extensions)]
    except:
        files = []
    files.sort()
    return [os.path.join(in_dir, f) for f in files]

def get_image_resolution_from_disk(filepath):
    img = None
    try:
        img = bpy.data.images.load(filepath, check_existing=False)
        try:
            w, h = int(img.size[0]), int(img.size[1])
        except:
            w, h = 0, 0
        return w, h
    except:
        return 0, 0
    finally:
        if img:
            safe_remove_datablock(bpy.data.images, img)

# ------------------------------------------------------------------------
# CORE FUNCTIONS
# ------------------------------------------------------------------------

def setup_bake_material(mesh_obj, source_image):
    mat = bpy.data.materials.new(name=_uid("TEMP_Batch_Mat"))
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()

    node_coord = nodes.new('ShaderNodeTexCoord')
    node_coord.location = (-1200, 300)

    node_tex = nodes.new('ShaderNodeTexImage')
    node_tex.image = source_image
    node_tex.extension = 'CLIP'
    node_tex.location = (-1000, 300)

    node_map = nodes.new('ShaderNodeVectorMath')
    node_map.operation = 'MULTIPLY_ADD'
    node_map.inputs[1].default_value = (2.0, 2.0, 2.0)
    node_map.inputs[2].default_value = (-1.0, -1.0, -1.0)
    node_map.location = (-700, 300)

    node_sep = nodes.new('ShaderNodeSeparateXYZ')
    node_sep.location = (-500, 300)

    node_inv_z = nodes.new('ShaderNodeMath')
    node_inv_z.operation = 'MULTIPLY'
    node_inv_z.inputs[1].default_value = -1.0
    node_inv_z.location = (-350, 150)

    node_comb = nodes.new('ShaderNodeCombineXYZ')
    node_comb.location = (-150, 300)

    node_trans = nodes.new('ShaderNodeVectorTransform')
    node_trans.vector_type = 'NORMAL'
    node_trans.convert_from = 'CAMERA'
    node_trans.convert_to = 'WORLD'
    node_trans.location = (50, 300)

    node_bsdf = nodes.new('ShaderNodeBsdfPrincipled')
    node_bsdf.location = (400, 0)

    node_out = nodes.new('ShaderNodeOutputMaterial')
    node_out.location = (700, 0)

    node_target = nodes.new('ShaderNodeTexImage')
    node_target.name = "BAKE_TARGET"
    node_target.location = (400, -300)

    links.new(node_coord.outputs['Window'], node_tex.inputs['Vector'])
    links.new(node_tex.outputs['Color'], node_map.inputs[0])
    links.new(node_map.outputs['Vector'], node_sep.inputs['Vector'])

    links.new(node_sep.outputs['X'], node_comb.inputs['X'])
    links.new(node_sep.outputs['Y'], node_comb.inputs['Y'])

    links.new(node_sep.outputs['Z'], node_inv_z.inputs[0])
    links.new(node_inv_z.outputs['Value'], node_comb.inputs['Z'])

    links.new(node_comb.outputs['Vector'], node_trans.inputs[0])
    links.new(node_trans.outputs['Vector'], node_bsdf.inputs['Normal'])
    links.new(node_bsdf.outputs['BSDF'], node_out.inputs['Surface'])

    nodes.active = node_target
    node_target.select = True

    mesh_obj.data.materials.clear()
    mesh_obj.data.materials.append(mat)

    return node_target, mat

def apply_culling(mesh_obj, camera_obj, angle_limit_rad, num_samples=1, tolerance=0.0001, expand_steps=1):
    mat_mask = bpy.data.materials.new(name=_uid("TEMP_Batch_Mask"))
    mat_mask.use_nodes = True
    try:
        bsdf = mat_mask.node_tree.nodes.get("Principled BSDF")
        if bsdf:
            bsdf.inputs['Base Color'].default_value = (0.0, 0.0, 0.0, 1.0)
    except:
        pass

    mesh_obj.data.materials.append(mat_mask)  # slot 1

    scene = bpy.context.scene
    mw = mesh_obj.matrix_world
    mw_norm = mesh_obj.matrix_world.inverted().transposed().to_3x3()
    cam_loc = camera_obj.matrix_world.translation
    depsgraph = bpy.context.evaluated_depsgraph_get()

    dot_threshold = -math.cos(angle_limit_rad)

    mesh = mesh_obj.data
    # Pre-transform all vertices to world space once (avoid per-poly per-vertex overhead)
    verts_world = [mw @ v.co for v in mesh.vertices]

    RAY_BIAS = 1e-4  # nudge ray origin slightly toward face to avoid self-intersection

    for poly in mesh.polygons:
        world_center = mw @ poly.center
        world_normal = (mw_norm @ poly.normal).normalized()

        direction = world_center - cam_loc
        dist_to_face = direction.length
        if dist_to_face <= 1e-8:
            poly.material_index = 1
            continue

        direction_norm = direction / dist_to_face

        # --- Back-face angle culling (unchanged) ---
        if world_normal.dot(direction_norm) > dot_threshold:
            poly.material_index = 1
            continue

        # --- Occlusion sampling ---
        # Always test center; if num_samples > 1 also test each vertex
        sample_pts = [world_center]
        if num_samples > 1:
            sample_pts += [verts_world[vi] for vi in poly.vertices]

        occluded_votes = 0
        total_votes = 0

        for pt in sample_pts:
            ray_dir = pt - cam_loc
            sample_dist = ray_dir.length
            if sample_dist <= 1e-8:
                continue

            ray_dir_norm = ray_dir / sample_dist
            # Offset origin by RAY_BIAS to avoid hitting the face itself at origin
            ray_origin = cam_loc + ray_dir_norm * RAY_BIAS

            hit, loc, _norm, _index, _hit_obj, _matrix = scene.ray_cast(
                depsgraph, ray_origin, ray_dir_norm
            )
            total_votes += 1

            if not hit:
                # No hit at all -> this sample direction is clear -> visible
                pass
            else:
                hit_dist = (loc - cam_loc).length
                # Absolute tolerance for rigid occlusion checking
                # Occluded if something was hit significantly CLOSER than the sample point
                if hit_dist < sample_dist - tolerance:
                    occluded_votes += 1

        if total_votes == 0:
            poly.material_index = 1
            continue

        # Majority vote: occluded if >50% of sample rays are blocked
        is_occluded = (occluded_votes / total_votes) > 0.5
        poly.material_index = 1 if is_occluded else 0

    if expand_steps > 0:
        import bmesh
        bm = bmesh.new()
        bm.from_mesh(mesh)
        bm.faces.ensure_lookup_table()

        for _ in range(expand_steps):
            to_mask = []
            for f in bm.faces:
                if f.material_index != 0:
                    continue
                # if any adjacent face is masked -> mask this one too
                touches_masked = False
                for e in f.edges:
                    for lf in e.link_faces:
                        if lf.material_index == 1:
                            touches_masked = True
                            break
                    if touches_masked:
                        break
                if touches_masked:
                    to_mask.append(f)

            for f in to_mask:
                f.material_index = 1

        bm.to_mesh(mesh)
        bm.free()
        mesh.update()

    return mat_mask

def create_and_assign_final_material(mesh_obj, texture_path):
    mat_name = "Baked_Normal_Material"
    if mat_name in bpy.data.materials:
        try:
            bpy.data.materials.remove(bpy.data.materials[mat_name])
        except:
            pass

    mat = bpy.data.materials.new(name=mat_name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()

    node_bsdf = nodes.new('ShaderNodeBsdfPrincipled')
    node_bsdf.location = (0, 0)

    node_out = nodes.new('ShaderNodeOutputMaterial')
    node_out.location = (300, 0)

    node_norm = nodes.new('ShaderNodeNormalMap')
    node_norm.location = (-300, -100)

    node_tex = nodes.new('ShaderNodeTexImage')
    node_tex.location = (-600, -100)

    if os.path.exists(texture_path):
        img = reimport_image_from_disk(texture_path)
        if img:
            node_tex.image = img

    links.new(node_tex.outputs['Color'], node_norm.inputs['Color'])
    links.new(node_norm.outputs['Normal'], node_bsdf.inputs['Normal'])
    links.new(node_tex.outputs['Alpha'], node_bsdf.inputs['Alpha'])
    links.new(node_bsdf.outputs['BSDF'], node_out.inputs['Surface'])

    mesh_obj.data.materials.clear()
    mesh_obj.data.materials.append(mat)
    mesh_obj.active_material_index = 0

def apply_normal_map_to_existing_material(mat, texture_path):
    if not mat:
        return False

    mat.use_nodes = True
    if not mat.node_tree:
        return False

    nt = mat.node_tree
    nodes = nt.nodes
    links = nt.links

    out_node = None
    for n in nodes:
        if n.type == 'OUTPUT_MATERIAL':
            out_node = n
            break

    principled = None
    if out_node:
        try:
            surf = out_node.inputs['Surface']
            if surf.is_linked and len(surf.links) > 0:
                fn = surf.links[0].from_node
                if fn and fn.type == 'BSDF_PRINCIPLED':
                    principled = fn
        except:
            pass

    if not principled:
        for n in nodes:
            if n.type == 'BSDF_PRINCIPLED':
                principled = n
                break

    if not principled:
        return False

    try:
        normal_in = principled.inputs['Normal']
        for lnk in list(normal_in.links):
            try:
                links.remove(lnk)
            except:
                pass
    except:
        pass

    img = None
    if texture_path and os.path.exists(texture_path):
        img = reimport_image_from_disk(texture_path)

    tex_node = nodes.get("BATCH_NORMAL_TEX")
    if not tex_node:
        tex_node = nodes.new('ShaderNodeTexImage')
        tex_node.name = "BATCH_NORMAL_TEX"
        tex_node.label = "BATCH_NORMAL_TEX"
        try:
            tex_node.location = (principled.location.x - 600, principled.location.y - 250)
        except:
            pass

    if img:
        tex_node.image = img
        try:
            tex_node.colorspace_settings.name = 'Non-Color'
        except:
            pass

        try:
            tex_node.extension = 'CLIP'
        except:
            pass

    norm_node = nodes.get("BATCH_NORMAL_MAP")
    if not norm_node:
        norm_node = nodes.new('ShaderNodeNormalMap')
        norm_node.name = "BATCH_NORMAL_MAP"
        norm_node.label = "BATCH_NORMAL_MAP"
        try:
            norm_node.location = (principled.location.x - 300, principled.location.y - 250)
        except:
            pass

    try:
        links.new(tex_node.outputs['Color'], norm_node.inputs['Color'])
    except:
        pass

    try:
        links.new(norm_node.outputs['Normal'], principled.inputs['Normal'])
    except:
        pass

    return True

def run_bake_cycle(context, mesh_obj, use_clear=True):
    scene = context.scene
    
    # ASSICURA MEDIA TYPE IMAGE PRIMA DEL BAKE
    try:
        scene.render.image_settings.media_type = 'IMAGE'
    except AttributeError:
        pass
        
    scene.render.engine = 'CYCLES'
    scene.cycles.bake_type = 'NORMAL'
    scene.render.bake.normal_space = 'TANGENT'
    scene.render.bake.use_clear = use_clear
    scene.render.bake.margin = 1

    ensure_active_selected(context, mesh_obj)
    bpy.ops.object.bake(type='NORMAL')

# ------------------------------------------------------------------------
# OPERATORS
# ------------------------------------------------------------------------

class OT_MatchRenderResolution(Operator):
    bl_idname = "object.match_render_resolution"
    bl_label = "Match Render Resolution"
    bl_description = "Set Output render resolution equal to the resolution of images in the Input folder and force format to PNG"

    def execute(self, context):
        props = context.scene.bake_settings
        in_dir = props.input_path

        if not in_dir or not os.path.exists(in_dir):
            self.report({'ERROR'}, "Invalid Input Folder")
            return {'CANCELLED'}

        paths = get_folder_images_sorted(in_dir)
        if not paths:
            self.report({'ERROR'}, "No images found in Input Folder")
            return {'CANCELLED'}

        first_path = paths[0]
        w0, h0 = get_image_resolution_from_disk(first_path)
        if w0 <= 0 or h0 <= 0:
            self.report({'ERROR'}, "Couldn't read resolution from first image")
            return {'CANCELLED'}

        mismatch = False
        for p in paths[1:]:
            w, h = get_image_resolution_from_disk(p)
            if w <= 0 or h <= 0:
                continue
            if (w != w0) or (h != h0):
                mismatch = True
                break

        context.scene.render.resolution_x = int(w0)
        context.scene.render.resolution_y = int(h0)

        # --- Force Media Type to Image (PNG) ---
        try:
            context.scene.render.image_settings.media_type = 'IMAGE'
        except AttributeError:
            pass
            
        try:
            context.scene.render.image_settings.file_format = 'PNG'
        except: pass

        if mismatch:
            self.report({'WARNING'}, "Images have different resolutions. Set render resolution to the first image.")
            show_message_box(
                "Le immagini nella cartella hanno risoluzioni diverse tra loro.\n"
                "Ho impostato la render resolution usando la prima immagine.",
                title="Attenzione",
                icon='ERROR'
            )
        else:
            self.report({'INFO'}, f"Render resolution set to {w0}x{h0} and format to PNG")

        return {'FINISHED'}

class OT_CreateCameraProjectionShader(Operator):
    bl_idname = "object.create_camera_projection_shader"
    bl_label = "Create Camera Projection Shader"
    bl_description = "Create and apply the same temporary camera-projection shader used during baking for the selected image/camera"

    def execute(self, context):
        props = context.scene.bake_settings
        mesh_obj = props.target_mesh

        if not mesh_obj:
            self.report({'ERROR'}, "Select Target Mesh!")
            return {'CANCELLED'}

        in_dir = bpy.path.abspath(props.input_path) if props.input_path else ""
        if not in_dir or not os.path.exists(in_dir):
            self.report({'ERROR'}, "Invalid Input Folder")
            return {'CANCELLED'}

        paths = get_folder_images_sorted(in_dir)
        if not paths:
            self.report({'ERROR'}, "No images found in Input Folder")
            return {'CANCELLED'}

        n = len(paths)
        idx = int(props.debug_image_index)
        if idx < 1:
            idx = 1
        if idx > n:
            idx = n

        img_path = paths[idx - 1]
        cam_name = os.path.splitext(os.path.basename(img_path))[0]
        cam_obj = bpy.data.objects.get(cam_name)

        if (not cam_obj) or cam_obj.type != 'CAMERA':
            self.report({'ERROR'}, f"No matching Camera found for image '{os.path.basename(img_path)}' (expected camera '{cam_name}')")
            return {'CANCELLED'}

        # Backup only once (do not overwrite on subsequent debug clicks).
        debug_backup_store_on_object(mesh_obj)

        scene = context.scene
        orig_camera = scene.camera

        # Set active camera to the chosen one, and enter camera view.
        scene.camera = cam_obj
        set_any_3d_view_to_camera_view()

        try:
            try:
                src_image = bpy.data.images.load(img_path, check_existing=False)
                try:
                    src_image.colorspace_settings.name = 'Non-Color'
                except:
                    pass
            except:
                self.report({'ERROR'}, f"Failed to load image: {os.path.basename(img_path)}")
                return {'CANCELLED'}

            ensure_active_selected(context, mesh_obj)
            setup_bake_material(mesh_obj, src_image)
            apply_culling(mesh_obj, cam_obj, props.angle_limit)

            self.report({'INFO'}, f"Temporary bake shader applied for {cam_obj.name}")
            return {'FINISHED'}
        finally:
            pass

class OT_RestorePreDebugMaterial(Operator):
    bl_idname = "object.restore_pre_debug_material"
    bl_label = "Restore Previous Material"
    bl_description = "Restore the target mesh material state saved before applying the camera projection debug shader"

    def execute(self, context):
        props = context.scene.bake_settings
        mesh_obj = props.target_mesh

        if not mesh_obj:
            self.report({'ERROR'}, "Select Target Mesh!")
            return {'CANCELLED'}

        ensure_active_selected(context, mesh_obj)

        ok = debug_backup_restore_from_object(mesh_obj)
        if ok:
            self.report({'INFO'}, "Previous material restored.")
            return {'FINISHED'}
        else:
            self.report({'WARNING'}, "No saved debug material state found.")
            return {'CANCELLED'}

class OT_BatchBakeNormals(Operator):
    bl_idname = "object.batch_bake_normals"
    bl_label = "Start Batch Bake"
    bl_description = "Process maps from Input folder to Output folder"

    def execute(self, context):
        props = context.scene.bake_settings
        mesh_obj = props.target_mesh

        if not mesh_obj:
            self.report({'ERROR'}, "Select Target Mesh!")
            return {'CANCELLED'}

        ensure_active_selected(context, mesh_obj)

        in_dir = props.input_path
        out_dir = props.output_path

        if not in_dir or not os.path.exists(in_dir):
            self.report({'ERROR'}, "Invalid Input Folder")
            return {'CANCELLED'}
        if not out_dir:
            self.report({'ERROR'}, "Missing Output Folder")
            return {'CANCELLED'}
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        valid_extensions = ('.jpg', '.jpeg', '.png', '.tga', '.exr', '.tif', '.tiff')
        files = [f for f in os.listdir(in_dir) if f.lower().endswith(valid_extensions)]

        missing = []
        for f_name in files:
            cam_name = os.path.splitext(f_name)[0]
            obj = bpy.data.objects.get(cam_name)
            if (not obj) or (obj.type != 'CAMERA'):
                missing.append(f_name)

        if missing:
            preview = ", ".join(missing[:10])
            more = "" if len(missing) <= 10 else f" (+{len(missing) - 10}...)"
            self.report({'WARNING'}, f"{len(missing)} image(s) without matching cameras found.")
            show_message_box(
                f"Trovate {len(missing)} immagini senza camera corrispondente in Blender.\n"
                f"Esempi: {preview}{more}\n"
                f"(Il bake proseguirà usando solo le coppie valide.)",
                title="Attenzione",
                icon='ERROR'
            )

        valid_pairs = []
        for f_name in files:
            cam_name = os.path.splitext(f_name)[0]
            obj = bpy.data.objects.get(cam_name)
            if obj and obj.type == 'CAMERA':
                valid_pairs.append((obj, os.path.join(in_dir, f_name)))

        if not valid_pairs:
            self.report({'ERROR'}, "No matching Cameras found.")
            return {'CANCELLED'}

        scene = context.scene
        
        # SALVATAGGIO STATO INIZIALE MEDIA TYPE
        old_media_type = getattr(scene.render.image_settings, "media_type", 'IMAGE')
        try:
            scene.render.image_settings.media_type = 'IMAGE'
        except AttributeError:
            pass

        # --- Inject blending here ---
        if props.blend_input_to_object_normals:
            try:
                valid_pairs = _prepare_blended_input_normals(context, mesh_obj, valid_pairs, out_dir, props)
            except Exception as e:
                self.report({'ERROR'}, f"Blend Input to Object Normals failed: {e}")
                return {'CANCELLED'}

        scene = context.scene
        orig_engine = scene.render.engine
        orig_camera = scene.camera
        orig_res_x = scene.render.resolution_x
        orig_res_y = scene.render.resolution_y
        orig_res_pct = scene.render.resolution_percentage
        scene.render.resolution_percentage = 100

        selection_state = backup_selection_state(context)
        mesh_state = backup_mesh_material_state(mesh_obj)

        ext_map = {'PNG': '.png', 'JPEG': '.jpg', 'TARGA': '.tga', 'OPEN_EXR': '.exr', 'TIFF': '.tif'}
        ext = ext_map.get(props.file_format, ".png")

        comp_image = None

        composite_path = os.path.join(out_dir, f"Composite_Normal_Map{ext}")
        composite_depth = props.color_depth if props.file_format in {'PNG', 'OPEN_EXR', 'TIFF'} else '8'

        try:
            if props.create_composite:
                comp_image = bpy.data.images.new(
                    name=_uid("TEMP_Composite_Buffer"),
                    width=props.res_x, height=props.res_y,
                    alpha=True, float_buffer=True
                )
                comp_image.colorspace_settings.name = 'Non-Color'
                comp_image.generated_color = (0.5, 0.5, 1.0, 1.0)

            for cam_obj, img_path in valid_pairs:

                ensure_active_selected(context, mesh_obj)
                scene.camera = cam_obj

                # --- Set Projection Resolution per Camera Image ---
                w_img, h_img = get_image_resolution_from_disk(img_path)
                if w_img > 0 and h_img > 0:
                    scene.render.resolution_x = int(w_img)
                    scene.render.resolution_y = int(h_img)

                src_image = None
                main_mat = None
                mat_mask = None
                single_bake_image = None
                node_target = None

                try:
                    try:
                        src_image = bpy.data.images.load(img_path, check_existing=False)
                        src_image.colorspace_settings.name = 'Non-Color'
                    except:
                        continue

                    node_target, main_mat = setup_bake_material(mesh_obj, src_image)
                    mat_mask = apply_culling(mesh_obj, cam_obj, props.angle_limit, getattr(props, 'occlusion_samples', 1), getattr(props, 'occlusion_tolerance', 0.0001), getattr(props, 'occlusion_expand_steps', 1))

                    single_bake_image = bpy.data.images.new(
                        name=_uid(f"TEMP_Bake_{cam_obj.name}"),
                        width=props.res_x, height=props.res_y,
                        alpha=True, float_buffer=True
                    )
                    single_bake_image.colorspace_settings.name = 'Non-Color'
                    single_bake_image.generated_color = (0.0, 0.0, 0.0, 0.0)
                    node_target.image = single_bake_image

                    run_bake_cycle(context, mesh_obj, use_clear=False)

                    file_name = f"normal_{cam_obj.name}_tangent{ext}"
                    full_path = os.path.join(out_dir, file_name)

                    ok = save_image_robust(single_bake_image, full_path, props.file_format, props.color_depth)
                    if not ok:
                        print(f"Error saving {file_name}")

                    if props.create_composite and comp_image:
                        node_target.image = comp_image
                        run_bake_cycle(context, mesh_obj, use_clear=False)

                finally:
                    if single_bake_image:
                        safe_remove_datablock(bpy.data.images, single_bake_image)
                    if src_image:
                        safe_remove_datablock(bpy.data.images, src_image)
                    if mat_mask:
                        safe_remove_datablock(bpy.data.materials, mat_mask)
                    if main_mat:
                        safe_remove_datablock(bpy.data.materials, main_mat)

            mesh_obj.data.materials.clear()
            bpy.ops.wm.redraw_timer(type='DRAW_WIN_SWAP', iterations=1)

            if props.create_composite and comp_image:
                ok = save_image_robust(comp_image, composite_path, props.file_format, composite_depth)
                if not ok:
                    composite_path = None
                else:
                    wait_for_file_ready(composite_path, timeout=12.0)

            if props.apply_to_mesh:
                tex_to_use = None

                if props.create_composite and composite_path and os.path.exists(composite_path):
                    tex_to_use = composite_path
                else:
                    if valid_pairs:
                        last_cam = valid_pairs[-1][0]
                        candidate = os.path.join(out_dir, f"normal_{last_cam.name}_tangent{ext}")
                        if os.path.exists(candidate):
                            tex_to_use = candidate

                if tex_to_use:
                    ensure_active_selected(context, mesh_obj)

                    if props.update_original_material:
                        if len(mesh_state["orig_mats"]) == 0:
                            create_and_assign_final_material(mesh_obj, tex_to_use)
                            self.report({'INFO'}, "Success! Material Applied.")
                        else:
                            restore_mesh_material_state(mesh_obj, mesh_state)
                            
                            mat = mesh_obj.active_material
                            if not mat:
                                for m in mesh_obj.data.materials:
                                    if m:
                                        mat = m
                                        break

                            if mat:
                                ok = apply_normal_map_to_existing_material(mat, tex_to_use)
                                if ok:
                                    self.report({'INFO'}, "Success! Original Material Updated.")
                                else:
                                    create_and_assign_final_material(mesh_obj, tex_to_use)
                                    self.report({'WARNING'}, "No Principled BSDF found; created a new material.")
                            else:
                                create_and_assign_final_material(mesh_obj, tex_to_use)
                                self.report({'INFO'}, "Success! Material Applied.")
                    else:
                        create_and_assign_final_material(mesh_obj, tex_to_use)
                        self.report({'INFO'}, "Success! Material Applied.")
                else:
                    self.report({'WARNING'}, "Bake done, but no texture found for material.")
            else:
                restore_mesh_material_state(mesh_obj, mesh_state)

        finally:
            
            # RIPRISTINO MEDIA TYPE
            try:
                scene.render.image_settings.media_type = old_media_type
            except AttributeError:
                pass
                
            scene.render.engine = orig_engine
            scene.camera = orig_camera

            if comp_image:
                safe_remove_datablock(bpy.data.images, comp_image)

            restore_selection_state(context, selection_state)

        return {'FINISHED'}


# ------------------------------------------------------------------------
# PANELS
# ------------------------------------------------------------------------


import numpy as np
import uuid

def compute_slope_correction(meshobj, cameraobj, input_img_path, output_img_path, props, debug_dir=None):
    save_debug = getattr(props, "save_debug_images", False)
    mesh = meshobj.data

    attr_name = "BNBP_FaceIndexRGB"
    if attr_name in mesh.attributes:
        mesh.attributes.remove(mesh.attributes[attr_name])

    attr = mesh.attributes.new(name=attr_name, type='FLOAT_COLOR', domain='FACE')
    count = len(mesh.polygons)
    colors = np.zeros((count, 4), dtype=np.float32)
    indices = np.arange(count, dtype=np.int32)

    colors[:, 0] = (indices % 256) / 255.0
    colors[:, 1] = ((indices // 256) % 256) / 255.0
    colors[:, 2] = ((indices // 65536) % 256) / 255.0
    colors[:, 3] = 1.0
    attr.data.foreach_set('color', colors.ravel())

    mat = bpy.data.materials.new(name="TEMP_FaceIndex")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()

    node_attr = nodes.new("ShaderNodeAttribute")
    node_attr.attribute_name = attr_name
    node_em = nodes.new("ShaderNodeEmission")
    node_out = nodes.new("ShaderNodeOutputMaterial")
    links.new(node_attr.outputs['Color'], node_em.inputs['Color'])
    links.new(node_em.outputs['Emission'], node_out.inputs['Surface'])

    orig_mats = [slot.material for slot in meshobj.material_slots]
    meshobj.data.materials.clear()
    meshobj.data.materials.append(mat)

    scene = bpy.context.scene
    orig_engine = scene.render.engine
    orig_camera = scene.camera
    orig_filepath = scene.render.filepath
    orig_view_transform = scene.view_settings.view_transform
    orig_film_transp = scene.render.film_transparent

    # Use CYCLES with BOX filter to completely disable Anti-Aliasing
    scene.render.engine = 'CYCLES'
    try:
        scene.cycles.samples = 1
        scene.cycles.use_denoising = False
        scene.cycles.max_bounces = 0
    except:
        pass
    try:
        scene.cycles.pixel_filter_type = 'BOX'
        scene.cycles.filter_width = 0.01
    except:
        pass

    scene.view_settings.view_transform = 'Raw'
    scene.render.film_transparent = True
    scene.camera = cameraobj

    img_in = reimport_image_from_disk(input_img_path)
    if not img_in:
        raise Exception(f"Cannot load {input_img_path}")

    w, h = img_in.size[0], img_in.size[1]

    orig_res_x = scene.render.resolution_x
    orig_res_y = scene.render.resolution_y
    orig_pct = scene.render.resolution_percentage
    scene.render.resolution_x = w
    scene.render.resolution_y = h
    scene.render.resolution_percentage = 100

    uid = uuid.uuid4().hex[:8]
    temp_render_path = os.path.join(bpy.app.tempdir, f"face_index_{uid}.exr")
    scene.render.filepath = temp_render_path

    orig_fmt = scene.render.image_settings.file_format
    orig_mode = scene.render.image_settings.color_mode
    orig_depth = scene.render.image_settings.color_depth

    scene.render.image_settings.file_format = 'OPEN_EXR'
    scene.render.image_settings.color_mode = 'RGBA'
    scene.render.image_settings.color_depth = '32'

    # Render Pass 1: Face Indices
    bpy.ops.render.render(write_still=True)

    img_idx = reimport_image_from_disk(temp_render_path)
    pixels_idx = np.empty(w * h * 4, dtype=np.float32)
    if img_idx:
        img_idx.pixels.foreach_get(pixels_idx)
    pixels_idx = pixels_idx.reshape((w * h, 4))

    if save_debug and debug_dir:
        import shutil
        try: shutil.copy(temp_render_path, os.path.join(debug_dir, f"{cameraobj.name}_face_index.exr"))
        except: pass
    if os.path.exists(temp_render_path):
        os.remove(temp_render_path)

    # Set to Non-Color data before reading
    try: img_in.colorspace_settings.name = 'Non-Color'
    except: pass

    pixels_in = np.empty(w * h * 4, dtype=np.float32)
    img_in.pixels.foreach_get(pixels_in)
    pixels_in = pixels_in.reshape((w * h, 4))

    R = np.round(pixels_idx[:, 0] * 255.0).astype(np.int32)
    G = np.round(pixels_idx[:, 1] * 255.0).astype(np.int32)
    B = np.round(pixels_idx[:, 2] * 255.0).astype(np.int32)
    A = pixels_idx[:, 3]

    triangle_indices = R + G * 256 + B * 65536
    triangle_indices = np.clip(triangle_indices, 0, count - 1)
    valid_mask = A > 0.5

    N_img = pixels_in[:, 0:3] * 2.0 - 1.0
    lengths_img = np.linalg.norm(N_img, axis=1, keepdims=True)
    lengths_img[lengths_img == 0] = 1.0
    N_img /= lengths_img

    valid_indices = triangle_indices[valid_mask]
    valid_normals = N_img[valid_mask]

    # 1. Flat Avg Normals
    sum_x = np.bincount(valid_indices, weights=valid_normals[:, 0], minlength=count)
    sum_y = np.bincount(valid_indices, weights=valid_normals[:, 1], minlength=count)
    sum_z = np.bincount(valid_indices, weights=valid_normals[:, 2], minlength=count)
    counts = np.bincount(valid_indices, minlength=count)

    counts_safe = np.maximum(counts, 1)
    avg_normals = np.column_stack((sum_x / counts_safe, sum_y / counts_safe, sum_z / counts_safe))
    lengths_avg = np.linalg.norm(avg_normals, axis=1, keepdims=True)
    lengths_avg[lengths_avg == 0] = 1.0
    avg_normals /= lengths_avg

    # --- Compute Smooth Avg Normals (vert_A) ---
    loops_vert_idx = np.empty(len(mesh.loops), dtype=np.int32)
    mesh.loops.foreach_get('vertex_index', loops_vert_idx)
    poly_loop_totals = np.empty(count, dtype=np.int32)
    mesh.polygons.foreach_get('loop_total', poly_loop_totals)
    poly_indices = np.repeat(np.arange(count), poly_loop_totals)

    A_vec_loops = avg_normals[poly_indices]
    V = len(mesh.vertices)
    vert_A = np.zeros((V, 3), dtype=np.float32)
    np.add.at(vert_A, loops_vert_idx, A_vec_loops)
    vert_A_lengths = np.linalg.norm(vert_A, axis=1, keepdims=True)
    vert_A_lengths[vert_A_lengths == 0] = 1.0
    vert_A /= vert_A_lengths

    colors_A = np.ones((V, 4), dtype=np.float32)
    colors_A[:, :3] = (vert_A + 1.0) * 0.5

    attr_A_name = "BNBP_SmoothAvg"
    if attr_A_name in mesh.attributes:
        mesh.attributes.remove(mesh.attributes[attr_A_name])
    attr_A = mesh.attributes.new(name=attr_A_name, type='FLOAT_COLOR', domain='POINT')
    attr_A.data.foreach_set('color', colors_A.ravel())

    # Create Mat A (Smooth Avg Normal)
    mat_A = bpy.data.materials.new(name="TEMP_MatA")
    mat_A.use_nodes = True
    nodes_A = mat_A.node_tree.nodes
    links_A = mat_A.node_tree.links
    nodes_A.clear()
    nA_attr = nodes_A.new("ShaderNodeAttribute")
    nA_attr.attribute_name = attr_A_name
    nA_em = nodes_A.new("ShaderNodeEmission")
    nA_out = nodes_A.new("ShaderNodeOutputMaterial")
    links_A.new(nA_attr.outputs['Color'], nA_em.inputs['Color'])
    links_A.new(nA_em.outputs['Emission'], nA_out.inputs['Surface'])

    # Create Mat B (Smooth View Space Mesh Normal)
    mat_B = bpy.data.materials.new(name="TEMP_MatB")
    mat_B.use_nodes = True
    nodes_B = mat_B.node_tree.nodes
    links_B = mat_B.node_tree.links
    nodes_B.clear()
    nB_geo = nodes_B.new("ShaderNodeNewGeometry")
    nB_trans = nodes_B.new("ShaderNodeVectorTransform")
    nB_trans.vector_type = 'NORMAL'
    nB_trans.convert_from = 'WORLD'
    nB_trans.convert_to = 'CAMERA'
    nB_sep = nodes_B.new("ShaderNodeSeparateXYZ")
    nB_comb = nodes_B.new("ShaderNodeCombineXYZ")
    nB_z = nodes_B.new("ShaderNodeMath")
    nB_z.operation = 'MULTIPLY'
    nB_z.inputs[1].default_value = -1.0
    nB_mul = nodes_B.new("ShaderNodeVectorMath")
    nB_mul.operation = 'MULTIPLY_ADD'
    nB_mul.inputs[1].default_value = (0.5, 0.5, 0.5)
    nB_mul.inputs[2].default_value = (0.5, 0.5, 0.5)
    nB_em = nodes_B.new("ShaderNodeEmission")
    nB_out = nodes_B.new("ShaderNodeOutputMaterial")

    links_B.new(nB_geo.outputs['Normal'], nB_trans.inputs['Vector'])
    links_B.new(nB_trans.outputs['Vector'], nB_sep.inputs['Vector'])
    links_B.new(nB_sep.outputs['X'], nB_comb.inputs['X'])
    links_B.new(nB_sep.outputs['Y'], nB_comb.inputs['Y'])
    links_B.new(nB_sep.outputs['Z'], nB_z.inputs[0])
    links_B.new(nB_z.outputs['Value'], nB_comb.inputs['Z'])
    links_B.new(nB_comb.outputs['Vector'], nB_mul.inputs[0])
    links_B.new(nB_mul.outputs['Vector'], nB_em.inputs['Color'])
    links_B.new(nB_em.outputs['Emission'], nB_out.inputs['Surface'])

    # Backup smooth states
    orig_smooth = np.empty(count, dtype=bool)
    mesh.polygons.foreach_get('use_smooth', orig_smooth)

    pixels_B_flat = None
    if save_debug and debug_dir:
        # Render Flat B (Mesh View Space Normals Flat)
        mesh.polygons.foreach_set('use_smooth', np.zeros(count, dtype=bool))
        mesh.update()

        temp_render_path_B_flat = os.path.join(bpy.app.tempdir, f"flat_B_{uid}.exr")
        meshobj.data.materials.clear()
        meshobj.data.materials.append(mat_B)
        scene.render.filepath = temp_render_path_B_flat
        bpy.ops.render.render(write_still=True)

        img_B_flat = reimport_image_from_disk(temp_render_path_B_flat)
        pixels_B_flat = np.empty(w * h * 4, dtype=np.float32)
        if img_B_flat: img_B_flat.pixels.foreach_get(pixels_B_flat)
        pixels_B_flat = pixels_B_flat.reshape((w * h, 4))

        if os.path.exists(temp_render_path_B_flat): os.remove(temp_render_path_B_flat)
        if img_B_flat: bpy.data.images.remove(img_B_flat)

    # Enforce Smooth Shading
    mesh.polygons.foreach_set('use_smooth', np.ones(count, dtype=bool))
    mesh.update()

    temp_render_path_A = os.path.join(bpy.app.tempdir, f"smooth_A_{uid}.exr")
    temp_render_path_B = os.path.join(bpy.app.tempdir, f"smooth_B_{uid}.exr")

    # Render Pass 2: A_vec Smooth
    meshobj.data.materials.clear()
    meshobj.data.materials.append(mat_A)
    scene.render.filepath = temp_render_path_A
    bpy.ops.render.render(write_still=True)
    img_A = reimport_image_from_disk(temp_render_path_A)
    pixels_A = np.empty(w * h * 4, dtype=np.float32)
    if img_A: img_A.pixels.foreach_get(pixels_A)
    pixels_A = pixels_A.reshape((w * h, 4))

    # Render Pass 3: B_vec Smooth
    meshobj.data.materials.clear()
    meshobj.data.materials.append(mat_B)
    scene.render.filepath = temp_render_path_B
    bpy.ops.render.render(write_still=True)
    img_B = reimport_image_from_disk(temp_render_path_B)
    pixels_B = np.empty(w * h * 4, dtype=np.float32)
    if img_B: img_B.pixels.foreach_get(pixels_B)
    pixels_B = pixels_B.reshape((w * h, 4))

    # Cleanup renders
    if os.path.exists(temp_render_path_A): os.remove(temp_render_path_A)
    if os.path.exists(temp_render_path_B): os.remove(temp_render_path_B)

    # Restore Scene & Mesh
    mesh.polygons.foreach_set('use_smooth', orig_smooth)
    mesh.update()

    meshobj.data.materials.clear()
    for m in orig_mats:
        meshobj.data.materials.append(m)

    scene.render.engine = orig_engine
    scene.view_settings.view_transform = orig_view_transform
    scene.render.film_transparent = orig_film_transp
    scene.camera = orig_camera
    scene.render.resolution_x = orig_res_x
    scene.render.resolution_y = orig_res_y
    scene.render.resolution_percentage = orig_pct
    scene.render.filepath = orig_filepath
    scene.render.image_settings.file_format = orig_fmt
    scene.render.image_settings.color_mode = orig_mode
    scene.render.image_settings.color_depth = orig_depth

    bpy.data.materials.remove(mat)
    bpy.data.materials.remove(mat_A)
    bpy.data.materials.remove(mat_B)

    if attr_name in mesh.attributes:
        mesh.attributes.remove(mesh.attributes[attr_name])
    if attr_A_name in mesh.attributes:
        mesh.attributes.remove(mesh.attributes[attr_A_name])

    # --- CALCULATE SMOOTH CORRECTION PER PIXEL ---
    A_smooth_img = pixels_A[:, 0:3] * 2.0 - 1.0
    B_smooth_img = pixels_B[:, 0:3] * 2.0 - 1.0

    A_smooth_valid = A_smooth_img[valid_mask]
    B_smooth_valid = B_smooth_img[valid_mask]

    len_A = np.linalg.norm(A_smooth_valid, axis=1, keepdims=True)
    len_A[len_A == 0] = 1.0
    A_smooth_valid /= len_A

    len_B = np.linalg.norm(B_smooth_valid, axis=1, keepdims=True)
    len_B[len_B == 0] = 1.0
    B_smooth_valid /= len_B

    v = np.cross(A_smooth_valid, B_smooth_valid)
    c = np.sum(A_smooth_valid * B_smooth_valid, axis=1)

    R_mat = np.zeros((len(valid_normals), 3, 3), dtype=np.float32)
    R_mat[:, 0, 0] = 1.0; R_mat[:, 1, 1] = 1.0; R_mat[:, 2, 2] = 1.0

    valid_pix = c > -0.999999
    v1, v2, v3 = v[valid_pix, 0], v[valid_pix, 1], v[valid_pix, 2]
    c_val = c[valid_pix]
    factor = 1.0 / (1.0 + c_val)

    vx = np.zeros((np.sum(valid_pix), 3, 3), dtype=np.float32)
    vx[:, 0, 1] = -v3; vx[:, 0, 2] = v2
    vx[:, 1, 0] = v3; vx[:, 1, 2] = -v1
    vx[:, 2, 0] = -v2; vx[:, 2, 1] = v1

    vx2 = np.matmul(vx, vx)
    R_mat[valid_pix] += vx + vx2 * factor[:, np.newaxis, np.newaxis]

    rotated_normals = np.einsum('nij,nj->ni', R_mat, valid_normals)
    r_lengths = np.linalg.norm(rotated_normals, axis=1, keepdims=True)
    r_lengths[r_lengths == 0] = 1.0
    rotated_normals /= r_lengths

    N_out = (rotated_normals + 1.0) * 0.5
    out_pixels = np.copy(pixels_in)
    out_pixels[valid_mask, 0:3] = N_out

    img_out = bpy.data.images.new(name="TEMP_SlopeCorr", width=w, height=h, alpha=True, float_buffer=True)
    try: img_out.colorspace_settings.name = 'Non-Color'
    except: pass
    img_out.pixels.foreach_set(out_pixels.ravel())

    os.makedirs(os.path.dirname(output_img_path), exist_ok=True)
    save_image_robust(img_out, output_img_path, getattr(props, 'file_format', 'PNG'), getattr(props, 'color_depth', '8'))

    if save_debug and debug_dir:
        img_avg_s = bpy.data.images.new(name="TEMP_AvgS", width=w, height=h, alpha=True, float_buffer=True)
        try: img_avg_s.colorspace_settings.name = 'Non-Color'
        except: pass

        # We overlay the smooth result on the transparent background from the input
        dbg_A_pixels = np.copy(pixels_in)
        dbg_A_pixels[valid_mask, 0:3] = pixels_A[valid_mask, 0:3]
        img_avg_s.pixels.foreach_set(dbg_A_pixels.ravel())
        save_image_robust(img_avg_s, os.path.join(debug_dir, f"{cameraobj.name}_avg_normal_smooth.png"), 'PNG', '16')
        bpy.data.images.remove(img_avg_s)

        img_mesh_s = bpy.data.images.new(name="TEMP_MeshS", width=w, height=h, alpha=True, float_buffer=True)
        try: img_mesh_s.colorspace_settings.name = 'Non-Color'
        except: pass

        dbg_B_pixels = np.copy(pixels_in)
        dbg_B_pixels[valid_mask, 0:3] = pixels_B[valid_mask, 0:3]
        img_mesh_s.pixels.foreach_set(dbg_B_pixels.ravel())
        save_image_robust(img_mesh_s, os.path.join(debug_dir, f"{cameraobj.name}_mesh_normal_smooth.png"), 'PNG', '16')
        bpy.data.images.remove(img_mesh_s)

        # Save avg flat
        img_avg_f = bpy.data.images.new(name="TEMP_AvgF", width=w, height=h, alpha=True, float_buffer=True)
        try: img_avg_f.colorspace_settings.name = 'Non-Color'
        except: pass
        dbg_A_f_pixels = np.copy(pixels_in)
        flat_A_colors = (avg_normals[valid_indices] + 1.0) * 0.5
        dbg_A_f_pixels[valid_mask, 0:3] = flat_A_colors
        img_avg_f.pixels.foreach_set(dbg_A_f_pixels.ravel())
        save_image_robust(img_avg_f, os.path.join(debug_dir, f"{cameraobj.name}_avg_normal_flat.png"), 'PNG', '16')
        bpy.data.images.remove(img_avg_f)

        # Save mesh flat
        if pixels_B_flat is not None:
            img_mesh_f = bpy.data.images.new(name="TEMP_MeshF", width=w, height=h, alpha=True, float_buffer=True)
            try: img_mesh_f.colorspace_settings.name = 'Non-Color'
            except: pass
            dbg_B_f_pixels = np.copy(pixels_in)
            dbg_B_f_pixels[valid_mask, 0:3] = pixels_B_flat[valid_mask, 0:3]
            img_mesh_f.pixels.foreach_set(dbg_B_f_pixels.ravel())
            save_image_robust(img_mesh_f, os.path.join(debug_dir, f"{cameraobj.name}_mesh_normal_flat.png"), 'PNG', '16')
            bpy.data.images.remove(img_mesh_f)

    if img_A: bpy.data.images.remove(img_A)
    if img_B: bpy.data.images.remove(img_B)
    bpy.data.images.remove(img_in)
    bpy.data.images.remove(img_out)
class OT_CorrectInputNormalsOperator(Operator):
    bl_idname = "object.correct_input_normals"
    bl_label = "Correct input normals based on mesh"
    bl_description = "Correct the input view space normal map slope based on target mesh geometry"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        props = context.scene.bake_settings
        meshobj = getattr(props, "target_mesh", None)
        if not meshobj:
            self.report({'ERROR'}, "Select Target Mesh!")
            return {'CANCELLED'}

        indir = getattr(props, "input_path", None)
        outdir = getattr(props, "output_path", None)
        if not indir or not os.path.exists(bpy.path.abspath(indir)):
            self.report({'ERROR'}, "Invalid Input Folder")
            return {'CANCELLED'}
        if not outdir:
            self.report({'ERROR'}, "Missing Output Folder")
            return {'CANCELLED'}

        indir = bpy.path.abspath(indir)
        outdir = bpy.path.abspath(outdir)

        corr_dir = os.path.join(outdir, "SlopeCorrected")
        os.makedirs(corr_dir, exist_ok=True)

        debug_dir = None
        if getattr(props, "save_debug_images", False):
            debug_dir = os.path.join(outdir, "SlopeCorrected_Debug")
            os.makedirs(debug_dir, exist_ok=True)

        valid_extensions = ('.jpg', '.jpeg', '.png', '.tga', '.exr', '.tif', '.tiff')
        files = [f for f in os.listdir(indir) if f.lower().endswith(valid_extensions)]

        valid_pairs = []
        for fname in files:
            cam_name = os.path.splitext(fname)[0]
            obj = bpy.data.objects.get(cam_name)
            if obj and obj.type == 'CAMERA':
                valid_pairs.append((obj, os.path.join(indir, fname), fname))

        if not valid_pairs:
            self.report({'ERROR'}, "No matching Cameras found for images in input folder.")
            return {'CANCELLED'}

        import time
        t0 = time.time()

        for camobj, in_path, fname in valid_pairs:
            out_path = os.path.join(corr_dir, fname)
            try:
                compute_slope_correction(meshobj, camobj, in_path, out_path, props, debug_dir)
            except Exception as e:
                self.report({'ERROR'}, f"Failed on {fname}: {str(e)}")

        if getattr(props, "use_corrected_map_as_new_input", False):
            props.input_path = corr_dir

        t1 = time.time()
        self.report({'INFO'}, f"Correction done in {t1-t0:.2f}s for {len(valid_pairs)} images.")
        return {'FINISHED'}


class PT_BatchBakePanel(Panel):
    bl_label = "Batch View to Tangent Normal Bake"
    bl_idname = "OBJECT_PT_batch_normal_bake_v2"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "NORM-v2t"
    bl_order = 0

    def draw(self, context):
        layout = self.layout
        props = context.scene.bake_settings
        
        box = layout.box()
        box.label(text="Target Mesh", icon='MESH_ICOSPHERE')
        box.prop(props, "target_mesh", text="Target Mesh")

        box = layout.box()
        box.label(text="Paths", icon='FILE_FOLDER')
        box.prop(props, "input_path", text="Input")
        box.prop(props, "output_path", text="Output")

        box.separator()
        box.operator("object.correct_input_normals", text="Correct Input Normals Based on Mesh", icon='NORMALS_VERTEX', depress=True)
        box.prop(props, "save_debug_images")
        box.prop(props, "use_corrected_map_as_new_input")
        box.separator()

        box = layout.box()
        box.label(text="Settings", icon='TEXTURE')
        col = box.column(align=True)
        col.prop(props, "res_x")
        col.prop(props, "res_y")
        col.prop(props, "file_format")
        if props.file_format in {'PNG', 'OPEN_EXR', 'TIFF'}:
            col.prop(props, "color_depth")

        box = layout.box()
        box.label(text="Pipeline", icon='NODETREE')

        box.prop(props, 'blend_input_to_object_normals')
        box.prop(props, "create_composite")
        box.prop(props, "apply_to_mesh")
        box.prop(props, "update_original_material")

        box.separator()
        
        row = box.row()
        icon = 'TRIA_DOWN' if props.show_advanced else 'TRIA_RIGHT'
        row.prop(props, "show_advanced", icon=icon, toggle=True, text="Advanced Parameters", icon_only=False, emboss=False)

        if props.show_advanced:
            adv_box = box.box()
            adv_box.prop(props, "angle_limit")
            adv_box.prop(props, "occlusion_samples")
            adv_box.prop(props, "occlusion_tolerance")
            adv_box.prop(props, "occlusion_expand_steps")

        layout.separator()
        layout.operator("object.batch_bake_normals", text="Start Batch Bake", icon='PLAY', depress=True)

class PT_BatchBakeDebugPanel(Panel):
    bl_label = "Debugging (Check Normals/Mesh Alignment)"
    bl_idname = "OBJECT_PT_batch_normal_bake_debug"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "NORM-v2t"
    bl_options = {'DEFAULT_CLOSED'}
    bl_order = 100

    def draw(self, context):
        layout = self.layout
        props = context.scene.bake_settings

        layout.operator("object.match_render_resolution", text="Match Render Resolution", icon='FULLSCREEN_ENTER')

        in_dir = bpy.path.abspath(props.input_path) if props.input_path else ""
        img_paths = get_folder_images_sorted(in_dir) if (in_dir and os.path.exists(in_dir)) else []
        n = len(img_paths)

        enabled = (n > 0)

        if enabled:
            try:
                if props.debug_image_index < 1:
                    props.debug_image_index = 1
                if props.debug_image_index > n:
                    props.debug_image_index = n
            except:
                pass

        row = layout.row(align=True)
        row.enabled = enabled
        row.prop(props, "debug_image_index", text="Normal/Camera #")
        row.label(text=f"/ {n}" if enabled else "/ 0")

        op_row = layout.row()
        op_row.enabled = enabled and (props.target_mesh is not None)
        op_row.operator(
            "object.create_camera_projection_shader",
            text="Create Camera Projection Shader",
            icon='MATERIAL'
        )

        restore_row = layout.row()
        has_backup = False
        try:
            if props.target_mesh and props.target_mesh.get("_BNBP_DEBUG_BACKUP", None):
                has_backup = True
        except:
            has_backup = False
        restore_row.enabled = (props.target_mesh is not None) and has_backup
        restore_row.operator(
            "object.restore_pre_debug_material",
            text="Restore Previous Material",
            icon='LOOP_BACK'
        )

# ------------------------------------------------------------------------
# REGISTER
# ------------------------------------------------------------------------

classes = (
    BakeSettings,
    OT_MatchRenderResolution,
    OT_CreateCameraProjectionShader,
    OT_RestorePreDebugMaterial,
    OT_BatchBakeNormals,
    OT_CorrectInputNormalsOperator,
    PT_BatchBakePanel,
    PT_BatchBakeDebugPanel
)

def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    bpy.types.Scene.bake_settings = PointerProperty(type=BakeSettings)

def unregister():
    del bpy.types.Scene.bake_settings
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)

if __name__ == "__main__":
    register()
