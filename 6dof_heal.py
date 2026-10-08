import mujoco
import numpy as np

def compute_forward_kinematics(xml_path, joint_angles, end_effector_name):
    # 1. Load the MuJoCo model and data
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)

    # 2. Set the 6 input joint parameters (ensure input is in radians)
    # This assumes the first 6 generalized coordinates belong to the arm
    data.qpos[:6] = joint_angles

    # 3. Compute forward kinematics
    mujoco.mj_kinematics(model, data)

    # 4. Get the ID of the end effector body (or site)
    try:
        ee_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, end_effector_name)
        if ee_id == -1:
            raise ValueError(f"Body '{end_effector_name}' not found. Check your XML for the correct name.")
    except Exception as e:
        print(e)
        return

    # 5. Extract Position (3x1) and Rotation Matrix (3x3)
    ee_pos = data.xpos[ee_id]
    ee_rot = data.xmat[ee_id].reshape(3, 3) # MuJoCo outputs a 9-element 1D array

    # 6. Construct the 4x4 Homogeneous Transformation Matrix
    T = np.eye(4)
    T[:3, :3] = ee_rot
    T[:3, 3] = ee_pos

    # 7. Output Results
    print("=== Forward Kinematics Results ===")
    print(f"Input Joint Angles (rad): {joint_angles}")
    print("\nEnd Effector Position (x, y, z):")
    print(np.round(ee_pos, 4))
    print("\n4x4 Transformation Matrix:")
    print(np.round(T, 4))

if __name__ == "__main__":
    # --- Configuration ---
    # Replace with the actual path to your main robot XML file (not just the arena)
    XML_FILE = "robot_descriptions/your_main_robot.xml" 
    
    # Replace with the name of the end effector body/link as defined in the XML
    EE_NAME = "end_effector_link" # Common names: 'ee_link', 'tool0', 'wrist_3_link'

    # The 6 joint parameters to tune (e.g., [q1, q2, q3, q4, q5, q6])
    Q_INPUTS = [0.0, -1.57, 1.57, 0.0, 1.57, 0.0]

    compute_forward_kinematics(XML_FILE, Q_INPUTS, EE_NAME)