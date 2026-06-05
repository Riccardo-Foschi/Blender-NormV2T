# Blender-NormV2T
NormV2T stands for (Normal View to Tangent). It is a plugin for Blender that project and merge multiple view-space-normal maps from specific camera position onto a single mesh

-----> [Download the Blender plugin from here!](https://github.com/Riccardo-Foschi/Blender-NormV2T/releases/download/v2.29/Norm-View2Tang_2-29.py)<-----

To install the plugin in Blender, go to Edit -> Preferences -> Addons -> Instal from disk
By installing the plugin (.py file) a new tab will appear in the sidebar (shortcut N)

<img width="317" height="905" alt="image" src="https://github.com/user-attachments/assets/d5d346f3-2e18-4990-af03-d1c911c9d903" />


Workflow:
- import a mesh (usually generated from photogrammetry)
- import any number of aligned cameras (these are the exact viewpoints where the view space normal maps refer to)
- with the picker set the target mesh onto which the view space normal maps will be projected
- select the input file folder where the normal maps are; for correct normal/camera matching, the normal map images must have the same names of their relative camera in Blender
- select the output file folder (where the new baked normal and debug maps will be saved)
- click the "Correct Input Normals Based on Mesh" button to generate new normal maps that are corrected by reading the real target mesh normals per pixel (this is particularly useful when the normals are created via photometric stereo pipelines, which are notoriously more biased compared to photogrammetry)
- by using "Save Debug Images and Use Corrected Maps as New Input" the corrected maps will be used for the next steps
- set the final texture resolution and format
- leave the "Blend Input to Object Normal" option unchecked if you corrected the input normals based on mesh
- leave  the "Generate Composite Map", "Apply Baked Normal to Mesh", and "Update Original MAterial" options checked to bake and reapply the new normal to the actual mesh while preserving the other maps already applied to the mesh
- the advanced parameters can be tweaked to reduce projection errors caused by face occlusion or reduce the angle threshold for the faces that will receive the projected normal
- click "Start Batch Bake" to project all the normal maps onto the mesh and create a composite map in the mesh UV space
- the Debugging panel allows to view the cameras with the corresponding normal to check alignment
