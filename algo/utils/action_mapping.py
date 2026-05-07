import numpy as np
from scipy.spatial.transform import Rotation as R
from roby.hardware.robots.fr3.robot_fr3 import FR3RobotAction, FR3ActionMode

def delta_position_action_mapping(self, action):
        if action is None:
            return FR3RobotAction(cartesian_positions=[0, 0, 0, 0, 0, 0, 1, 0], action_mode=FR3ActionMode.DELTA)
        if len(action) != 7:
            raise ValueError(f"POSITION_DELTA expects length 7 (xyz + rpy_deg + gripper), got {len(action)}")
        translation = np.array(action[:3])
        rotation = np.array(action[3:-1])
        rotation = R.from_euler("xyz", rotation, degrees=True).as_quat()
        gripper = np.array(action[-1])
        positions = np.concatenate([translation, rotation, [gripper]]).tolist()
        return FR3RobotAction(cartesian_positions=positions, action_mode=FR3ActionMode.DELTA)

def delta_joint_action_mapping(self, action):
    if action is None:
        return FR3RobotAction(cartesian_positions=[0, 0, 0, 0, 0, 0, 1, 0], action_mode=FR3ActionMode.DELTA)
    if len(action) != 8:
        raise ValueError(f"Expected action length 8, got {len(action)}")
    joint_position = np.array(action)
    return FR3RobotAction(joint_positions=joint_position, action_mode=FR3ActionMode.DELTA)

def absolute_position_action_mapping(self, action):
    if action is None:
        return FR3RobotAction(cartesian_positions=[0, 0, 0, 0, 0, 0, 1, 0], action_mode=FR3ActionMode.ABSOLUTE)
    if len(action) != 7:
        raise ValueError(f"POSITION_DELTA expects length 7 (xyz + rpy_deg + gripper), got {len(action)}")
    translation = np.array(action[:3])
    rotation = np.array(action[3:-1])
    rotation = R.from_euler("xyz", rotation, degrees=True).as_quat()
    gripper = np.array(action[-1])
    positions = np.concatenate([translation, rotation, [gripper]]).tolist()
    return FR3RobotAction(cartesian_positions=positions, action_mode=FR3ActionMode.ABSOLUTE)

def absolute_joint_action_mapping(self, action):
    if action is None:
        return FR3RobotAction(cartesian_positions=[0, 0, 0, 0, 0, 0, 1, 0], action_mode=FR3ActionMode.ABSOLUTE)
    if len(action) != 8:
        raise ValueError(f"Expected action length 8, got {len(action)}")
    joint_position = np.array(action)
    return FR3RobotAction(joint_positions=joint_position, action_mode=FR3ActionMode.ABSOLUTE)

# def delta_absolute_position_action_mapping(self, action):
#     if action is None:
#         return FR3RobotAction(cartesian_positions=[0, 0, 0, 0, 0, 0, 0], action_mode=FR3ActionMode.ABSOLUTE)
#     if len(action) != 7:
#         raise ValueError(f"POSITION_DELTA expects length 7 (xyz + rpy_deg + gripper), got {len(action)}")
    
#     state = self.robot.read_state()
#     tcp_state = np.array(state.end_effector_position)
    
#     print("tcp_state", tcp_state)
#     print("actions", action)
    
#     # 计算新的位置
#     translation = np.array(action[:3]) + tcp_state[:3]
#     rotation = np.array([1,0,0,0])
    
#     # 获取策略要求的夹爪目标状态（0或1）
#     target_state = action[-1]
    
#     # 只有当目标状态与当前状态不同时，才发送夹爪动作
#     gripper_cmd = 0  # 默认保持当前状态
    
#     if target_state != self.current_gripper_state:
#         if target_state == 1:
#             # 需要关闭夹爪（从开到关）
#             gripper_cmd = 1
#             self.current_gripper_state = 1
#             print("Closing gripper (open -> closed)")
#         elif target_state == 0:
#             # 需要打开夹爪（从关到开）
#             gripper_cmd = -1
#             self.current_gripper_state = 0
#             print("Opening gripper (closed -> open)")
#     else:
#         gripper_cmd = 0
    
#     positions = np.concatenate([translation, rotation, [gripper_cmd]]).tolist()
#     print("positions", positions)
    
#     return FR3RobotAction(cartesian_positions=positions, action_mode=FR3ActionMode.ABSOLUTE)

# def delta_absolute_position_action_mapping(self, action):
#     if action is None:
#         return FR3RobotAction(cartesian_positions=[0, 0, 0, 0, 0, 0, 0], action_mode=FR3ActionMode.ABSOLUTE)
#     if len(action) != 7:
#         raise ValueError(f"POSITION_DELTA expects length 7 (xyz + rpy_deg + gripper), got {len(action)}")
#     state = self.robot.read_state()
#     tcp_state = np.array(state.end_effector_position)
#     print("tcp_state", tcp_state)
#     print("actions", action)
#     translation = np.array(action[:3]) + tcp_state[:3]
#     rotation = np.array(action[3:-1]) + R.from_quat(tcp_state[[4, 5, 6, 3]]).as_euler("zyx", degrees=False)
#     rotation = R.from_euler("zyx", rotation, degrees=False).as_quat()[[3, 0, 1, 2]]
    
#     # 获取策略要求的夹爪目标状态（0或1）
#     target_state = action[-1]
    
#     # 只有当目标状态与当前状态不同时，才发送夹爪动作
#     gripper_cmd = 0  # 默认保持当前状态
    
#     if target_state != self.current_gripper_state:
#         if target_state == 1:
#             # 需要关闭夹爪（从开到关）
#             gripper_cmd = 1
#             self.current_gripper_state = 1
#             print("Closing gripper (open -> closed)")
#         elif target_state == 0:
#             # 需要打开夹爪（从关到开）
#             gripper_cmd = -1
#             self.current_gripper_state = 0
#             print("Opening gripper (closed -> open)")
#     else:
#         gripper_cmd = 0
    
#     positions = np.concatenate([translation, rotation, [gripper_cmd]]).tolist()
#     print("positions", positions)

#     return FR3RobotAction(cartesian_positions=positions, action_mode=FR3ActionMode.ABSOLUTE)

# def delta_absolute_position_action_mapping(self, action):
#     if action is None:
#         return FR3RobotAction(cartesian_positions=[0, 0, 0, 0, 0, 0, 0], action_mode=FR3ActionMode.ABSOLUTE)
    
#     if len(action) != 7:
#         raise ValueError(f"POSITION_DELTA expects length 7 (xyz + rpy_deg + gripper), got {len(action)}")
    
#     state = self.robot.read_state()
#     tcp_state = np.array(state.end_effector_position)
    
#     # 位置增量
#     translation = np.array(action[:3]) + tcp_state[:3]
    
#     # 旋转增量 - 使用四元数转欧拉角
#     # tcp_state中的四元数可能是 [w, x, y, z] 格式
#     # 转换为 [x, y, z, w] 格式供scipy使用

#     # 将当前四元数转换为欧拉角 (roll, pitch, yaw) = (X, Y, Z) 旋转
#     current_euler = R.from_quat(tcp_state[[4, 5, 6, 3]]).as_euler("zyx", degrees=False)  # 使用xyz顺序，因为你说绕X轴旋转
    
#     # 添加增量
#     # 注意：action[3:-1] 应该是 [roll, pitch, yaw] 增量
#     new_euler = np.array(action[3:-1]) + current_euler
    
#     # 将新的欧拉角转换回四元数
#     # 使用 'zyx' 顺序，因为X轴是第一个旋转轴（如果你要绕X轴旋转）
#     new_quat = R.from_euler("zyx", new_euler, degrees=False).as_quat()  # 返回 [x, y, z, w] 格式
    
#     # 将四元数转换回 [w, x, y, z] 格式（如果需要）
#     rotation = new_quat[[3, 0, 1, 2]]  # [w, x, y, z] 格式
    
#     # 获取策略要求的夹爪目标状态（0或1）
#     target_state = action[-1]
    
#     # 只有当目标状态与当前状态不同时，才发送夹爪动作
#     gripper_cmd = 0  # 默认保持当前状态
    
#     if target_state != self.current_gripper_state:
#         if target_state == 1:
#             # 需要关闭夹爪（从开到关）
#             gripper_cmd = 1
#             self.current_gripper_state = 1
#             print("Closing gripper (open -> closed)")
#         elif target_state == 0:
#             # 需要打开夹爪（从关到开）
#             gripper_cmd = -1
#             self.current_gripper_state = 0
#             print("Opening gripper (closed -> open)")
#     else:
#         gripper_cmd = 0
    
#     positions = np.concatenate([translation, rotation, [gripper_cmd]]).tolist()

#     return FR3RobotAction(cartesian_positions=positions, action_mode=FR3ActionMode.ABSOLUTE)

# def delta_absolute_position_action_mapping(self, action):
#     if action is None:
#         return FR3RobotAction(cartesian_positions=[0, 0, 0, 0, 0, 0, 0], action_mode=FR3ActionMode.ABSOLUTE)
    
#     if len(action) != 8:
#         raise ValueError(f"POSITION_DELTA expects length 7 (xyz + rpy_deg + gripper), got {len(action)}")
    
#     state = self.robot.read_state()
#     tcp_state = np.array(state.end_effector_position)
    
#     # 位置增量
#     translation = np.array(action[:3]) + tcp_state[:3]
    
#     # 旋转增量 - 使用旋转矩阵计算
#     # 当前姿态的四元数 (假设tcp_state[3:7]是[w, x, y, z]格式)
#     current_quat = tcp_state[3:7]  # [w, x, y, z]格式
    
#     # 转换为scipy接受的[x, y, z, w]格式
#     current_quat_scipy = np.array([current_quat[1], current_quat[2], current_quat[3], current_quat[0]])
    
#     # 获取当前旋转矩阵
#     current_rot = R.from_quat(current_quat_scipy)
#     current_rot_matrix = current_rot.as_matrix()
    
#     # 将增量欧拉角转换为旋转矩阵（相对于工具坐标系）
#     delta_euler = action[3:7]
#     delta_rot = R.from_quat(delta_euler)  # 使用xyz顺序
#     delta_rot_matrix = delta_rot.as_matrix()
    
#     # 计算新的旋转矩阵：当前旋转 × 增量旋转（相对于工具坐标系的旋转）
#     new_rot_matrix = current_rot_matrix @ delta_rot_matrix
    
#     # 将新的旋转矩阵转换为四元数
#     new_rot = R.from_matrix(new_rot_matrix)
#     new_quat_scipy = new_rot.as_quat()  # [x, y, z, w]格式
    
#     # 转换回[w, x, y, z]格式
#     rotation = [new_quat_scipy[3], new_quat_scipy[0], new_quat_scipy[1], new_quat_scipy[2]]
    
#     # 获取策略要求的夹爪目标状态（0或1）
#     target_state = action[-1]
    
#     # 只有当目标状态与当前状态不同时，才发送夹爪动作
#     gripper_cmd = 0  # 默认保持当前状态
    
#     if target_state != self.current_gripper_state:
#         if target_state == 1:
#             # 需要关闭夹爪（从开到关）
#             gripper_cmd = 1
#             self.current_gripper_state = 1
#             print("Closing gripper (open -> closed)")
#         elif target_state == 0:
#             # 需要打开夹爪（从关到开）
#             gripper_cmd = -1
#             self.current_gripper_state = 0
#             print("Opening gripper (closed -> open)")
#     else:
#         gripper_cmd = 0
    
#     positions = np.concatenate([translation, rotation, [gripper_cmd]]).tolist()

#     return FR3RobotAction(cartesian_positions=positions, action_mode=FR3ActionMode.ABSOLUTE)

# def delta_absolute_position_action_mapping(self, action):
#     if action is None:
#         return FR3RobotAction(cartesian_positions=[0, 0, 0, 0, 0, 0, 0], action_mode=FR3ActionMode.ABSOLUTE)
    
#     if len(action) != 8:
#         raise ValueError(f"POSITION_DELTA expects length 8 (xyz + rpy_deg + gripper), got {len(action)}")
    
#     state = self.robot.read_state()
#     tcp_state = np.array(state.end_effector_position)
    
#     # 位置增量（世界坐标系）
#     translation = np.array(action[:3]) + tcp_state[:3]
    
#     # 旋转处理 - 世界坐标系下的增量
#     # 当前姿态的四元数 (假设tcp_state[3:7]是[w, x, y, z]格式)
#     current_quat = tcp_state[3:7]  # [w, x, y, z]格式
    
#     # 转换为scipy接受的[x, y, z, w]格式
#     current_quat_scipy = np.array([current_quat[1], current_quat[2], current_quat[3], current_quat[0]])
    
#     # 获取当前旋转矩阵（世界坐标系）
#     current_rot = R.from_quat(current_quat_scipy)
    
#     # 将增量欧拉角（世界坐标系）转换为旋转矩阵
#     # 注意：这里假设action[3:6]是绕世界坐标系XYZ轴的欧拉角（弧度或度）
#     # 您需要根据实际情况确定是弧度还是度
#     delta_euler = action[3:6]  # 注意：这里取3:6，不是3:7
    
#     # 假设delta_euler是弧度，使用XYZ顺序
#     # 如果是度，需要转换为弧度：delta_euler_rad = np.radians(delta_euler)
#     delta_euler_rad = delta_euler  # 如果输入是弧度
#     # 或者：delta_euler_rad = np.radians(delta_euler)  # 如果输入是度
    
#     # 创建增量旋转（相对于世界坐标系）
#     delta_rot = R.from_euler('xyz', delta_euler_rad)
    
#     # 计算新的旋转：先应用增量旋转，再应用当前旋转
#     # 对于世界坐标系增量：new_rot = delta_rot * current_rot
#     new_rot = delta_rot * current_rot
    
#     # 将新的旋转转换为四元数
#     new_quat_scipy = new_rot.as_quat()  # [x, y, z, w]格式
    
#     # 转换回[w, x, y, z]格式
#     rotation = [new_quat_scipy[3], new_quat_scipy[0], new_quat_scipy[1], new_quat_scipy[2]]
    
#     # 处理夹爪动作
#     target_state = action[-1]
#     gripper_cmd = 0
    
#     if target_state != self.current_gripper_state:
#         if target_state == 1:
#             gripper_cmd = 1
#             self.current_gripper_state = 1
#             print("Closing gripper (open -> closed)")
#         elif target_state == 0:
#             gripper_cmd = -1
#             self.current_gripper_state = 0
#             print("Opening gripper (closed -> open)")
#     else:
#         gripper_cmd = 0
    
#     # 注意：这里只包含7个位置值（3位置 + 4四元数）
#     positions = np.concatenate([translation, rotation, [gripper_cmd]]).tolist()

#     return FR3RobotAction(cartesian_positions=positions, action_mode=FR3ActionMode.ABSOLUTE)

def delta_absolute_position_action_mapping(self, action):
    if action is None:
        return FR3RobotAction(cartesian_positions=[0, 0, 0, 0, 0, 0, 0], action_mode=FR3ActionMode.ABSOLUTE)
    
    if len(action) != 8:
        raise ValueError(f"POSITION_DELTA expects length 8 (xyz + rotation + gripper), got {len(action)}")
    
    state = self.robot.read_state()
    tcp_state = np.array(state.end_effector_position)
    
    # 位置增量（世界坐标系）
    translation = np.array(action[:3]) + tcp_state[:3]
    
    # 旋转处理 - 工具坐标系下的增量
    # 当前姿态的四元数 (假设tcp_state[3:7]是[w, x, y, z]格式)
    current_quat = tcp_state[3:7]  # [w, x, y, z]格式
    
    # 转换为scipy接受的[x, y, z, w]格式
    current_quat_scipy = np.array([current_quat[1], current_quat[2], current_quat[3], current_quat[0]])
    
    # 获取当前旋转矩阵（世界坐标系）
    current_rot = R.from_quat(current_quat_scipy)
    
    # 根据action的格式，旋转部分可能是四元数或欧拉角
    # 假设action[3:7]是四元数增量[x, y, z, w]（与merge函数输出一致）
    # 或者action[3:6]是欧拉角增量（弧度）
    
    if len(action) >= 7 and action[6] == 0:  # 第7个元素是占位符0，表示前6个元素是位置+欧拉角
        # 处理欧拉角增量
        delta_euler = action[3:6]  # 欧拉角增量
        # 假设delta_euler是弧度，使用XYZ顺序
        # 如果是度，需要转换为弧度：delta_euler_rad = np.radians(delta_euler)
        delta_euler_rad = delta_euler  # 如果输入是弧度
        delta_rot = R.from_euler('xyz', delta_euler_rad)
    else:
        # 处理四元数增量
        delta_quat_xyzw = action[3:7]  # 四元数增量[x, y, z, w]
        delta_rot = R.from_quat(delta_quat_xyzw)
    
    # 计算新的旋转：在当前工具坐标系下应用增量旋转
    # 对于工具坐标系增量：new_rot = current_rot * delta_rot（右乘）
    new_rot = current_rot * delta_rot
    
    # 将新的旋转转换为四元数
    new_quat_scipy = new_rot.as_quat()  # [x, y, z, w]格式
    
    # 转换回[w, x, y, z]格式
    rotation = [new_quat_scipy[3], new_quat_scipy[0], new_quat_scipy[1], new_quat_scipy[2]]
    
    # 处理夹爪动作
    target_state = action[-1]
    gripper_cmd = 0
    
    if target_state != self.current_gripper_state:
        if target_state == 1:
            gripper_cmd = 1
            self.current_gripper_state = 1
            print("Closing gripper (open -> closed)")
        elif target_state == 0:
            gripper_cmd = -1
            self.current_gripper_state = 0
            print("Opening gripper (closed -> open)")
    else:
        gripper_cmd = 0
    
    # 注意：这里只包含7个位置值（3位置 + 4四元数）
    positions = np.concatenate([translation, rotation, [gripper_cmd]]).tolist()

    return FR3RobotAction(cartesian_positions=positions, action_mode=FR3ActionMode.ABSOLUTE)