import mujoco
import numpy as np

def compute_franka_fk(xml_path, joint_angles, ee_name="panda_hand"):
    """
    Computes Forward Kinematics for a 7-DOF Franka Emika Panda arm.
    
    :param xml_path: Path to the Franka Panda XML file.
    :param joint_angles: List or array of 7 joint angles in radians.
    :param ee_name: Name of the end-effector body or site (default: 'panda_hand').
    """
    if len(joint_angles) != 7:
        raise ValueError(f"Expected 7 joint angles for Franka, got {len(joint_angles)}")

    # 1. Load the MuJoCo model and data
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)

    # 2. Set the 7 joint positions
    data.qpos[:7] = joint_angles

    # 3. Compute forward kinematics
    mujoco.mj_kinematics(model, data)

    # 4. Find the End-Effector ID (checking both body and site)
    ee_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, ee_name)
    is_site = False

    if ee_id == -1:
        # Try finding as a site if body lookup fails
        ee_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, ee_name)
        is_site = True

    if ee_id == -1:
        raise ValueError(
            f"Could not find '{ee_name}' as a body or site in the XML model."
        )

    # 5. Extract End-Effector Position and Rotation
    if is_site:
        ee_pos = data.site_xpos[ee_id]
        ee_rot = data.site_xmat[ee_id].reshape(3, 3)
    else:
        ee_pos = data.xpos[ee_id]
        ee_rot = data.xmat[ee_id].reshape(3, 3)

    # 6. Build 4x4 Homogeneous Transformation Matrix T
    T = np.eye(4)
    T[:3, :3] = ee_rot
    T[:3, 3] = ee_pos

    return ee_pos, T

if __name__ == "__main__":
    # Path to your Franka Panda XML file
    XML_FILE = "franka_emika_panda/panda.xml"

    # End-effector name (common names in Franka models: 'panda_hand', 'panda_link8', or 'attachment_site')
    EE_NAME = "panda_hand"

    # 7-DOF Joint configuration (angles in radians)
    # [joint1, joint2, joint3, joint4, joint5, joint6, joint7]
    Q_INPUTS = [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785]  # Standard home configuration

    pos, transform_matrix = compute_franka_fk(XML_FILE, Q_INPUTS, EE_NAME)

    print("=== 7-DOF Franka Emika Panda FK Results ===")
    print(f"Joint Angles (rad): {Q_INPUTS}")
    print("\nEnd-Effector Position [x, y, z] (m):")
    print(np.round(pos, 4))
    print("\n4x4 Transformation Matrix (T):")
    print(np.round(transform_matrix, 4))