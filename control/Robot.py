import numpy as np
import pybullet as p
# Robot with Camera Class
class eye_in_hand_robot:
    def get_ee_position(self):
        '''
        Function to return the end-effector of the link. This is the very tip of the robot at the end of the jaws.
        '''
        endEffectorIndex = self.numActiveJoints
        endEffectorState = p.getLinkState(self.robot_id, endEffectorIndex)
        endEffectorPos = np.array(endEffectorState[0])
        endEffectorOrn = np.array(p.getMatrixFromQuaternion(endEffectorState[1])).reshape(3,3)
        
        #add an offset to get past the forceps
        endEffectorPos += self.camera_offset*endEffectorOrn[:,2]
        return endEffectorPos, endEffectorOrn

    def get_current_joint_angles(self):
        # Get the current joint angles
        joint_angles = np.zeros(self.numActiveJoints)
        for i in range(self.numActiveJoints):
            joint_state = p.getJointState(self.robot_id, self._active_joint_indices[i])
            joint_angles[i] = joint_state[0]
        return joint_angles
    
    def get_jacobian_at_current_position(self):
        #Returns the Robot Jacobian of the last active link
        mpos, mvel, mtorq = self.get_active_joint_states()   
        zero_vec = [0.0]*len(mpos)
        linearJacobian, angularJacobian = p.calculateJacobian(self.robot_id, 
                                                              self.numActiveJoints,
                                                              [0,0,self.camera_offset],
                                                              mpos, 
                                                              zero_vec,
                                                              zero_vec)
        #only return the active joint's jacobians
        Jacobian = np.vstack((linearJacobian,angularJacobian))
        return Jacobian[:,:self.numActiveJoints]
    
    def set_joint_position(self, desireJointPositions, kp=1.0, kv=0.3):
        '''Set  the joint angle positions of the robot'''
        zero_vec = [0.0] * self._numLinkJoints
        allJointPositionObjectives = [0.0]*self._numLinkJoints
        for i in range(desireJointPositions.shape[0]):
            idx = self._active_joint_indices[i]
            allJointPositionObjectives[idx] = desireJointPositions[i]

        p.setJointMotorControlArray(self.robot_id,
                                    range(self._numLinkJoints),
                                    p.POSITION_CONTROL,
                                    targetPositions=allJointPositionObjectives,
                                    targetVelocities=zero_vec,
                                    positionGains=[kp] * self._numLinkJoints,
                                    velocityGains=[kv] * self._numLinkJoints)

    def get_active_joint_states(self):
        '''Get the states (position, velocity, and torques) of the active joints of the robot
        '''
        joint_states = p.getJointStates(self.robot_id, range(self._numLinkJoints))
        joint_infos = [p.getJointInfo(self.robot_id, i) for i in range(self._numLinkJoints)]
        joint_states = [j for j, i in zip(joint_states, joint_infos) if i[3] > -1]
        joint_positions = [state[0] for state in joint_states]
        joint_velocities = [state[1] for state in joint_states]
        joint_torques = [state[3] for state in joint_states]
        return joint_positions, joint_velocities, joint_torques


         
    def __init__(self, robot_id, initialJointPos):
        self.robot_id = robot_id
        self.eeFrameId = []
        self.camera_offset = 0.1 #offset camera in z direction to avoid grippers
        # Get the joint info
        self._numLinkJoints = p.getNumJoints(self.robot_id) #includes passive joint
        jointInfo = [p.getJointInfo(self.robot_id, i) for i in range(self._numLinkJoints)]
        
        # Get joint locations (some joints are passive)
        self._active_joint_indices = []
        for i in range(self._numLinkJoints):
            if jointInfo[i][2]==p.JOINT_REVOLUTE:
                self._active_joint_indices.append(jointInfo[i][0])
        self.numActiveJoints = len(self._active_joint_indices) #exact number of active joints

        #reset joints
        for i in range(self._numLinkJoints):
            p.resetJointState(self.robot_id,i,initialJointPos[i])