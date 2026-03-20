# quadruped URDF

Place your URDF file here after exporting from FreeCAD.

## Export steps from FreeCAD:
1. Export each body part as STL to robots/quadruped/meshes/
2. Write robot.urdf referencing those mesh files
3. Convert to MJCF: python -c "import mujoco; m = mujoco.MjModel.from_xml_path('robot.urdf')"
4. Copy the converted XML to robots/quadruped/model.xml

## URDF to MJCF conversion:
    python -c "
    import mujoco
    model = mujoco.MjModel.from_xml_path('robots/quadruped/urdf/robot.urdf')
    with open('robots/quadruped/model_from_urdf.xml', 'w') as f:
        f.write(mujoco.mj_saveXML(model))
    print('Converted successfully')
    "
