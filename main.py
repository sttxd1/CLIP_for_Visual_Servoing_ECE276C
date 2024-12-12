#import matplotlib.pyplot as plt
import numpy as np
import pybullet as p
import pybullet_data
import os
import time
import cv2
from control.funcs import *
from control.Robot import eye_in_hand_robot
from control.uv_filter import uv_filter
import imageio.v2 as imageio
import matplotlib.pyplot as plt
import torch
import clip
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)
model, preprocess = clip.load("ViT-B/32", device=device)

def encode_text(text):
    text_tokens = clip.tokenize([text]).to(device)
    with torch.no_grad():
        text_features = model.encode_text(text_tokens)
        text_features /= text_features.norm(dim=-1, keepdim=True)
    return text_features

def encode_image(pil_img):
    image_input = preprocess(pil_img).unsqueeze(0).to(device)
    with torch.no_grad():
        image_features = model.encode_image(image_input)
        image_features /= image_features.norm(dim=-1, keepdim=True)
    return image_features

#camera (don't change these settings)
camera_width = 512                                             #image width
camera_height = 512                                            #image height
camera_fov = 120                                                #field of view of camera
camera_focal_depth = 0.5*camera_height/np.tan(0.5*np.pi/180*camera_fov) 
                                                               #focal depth in pixel space
camera_aspect = camera_width/camera_height                     #aspect ratio
camera_near = 0.02                                             #near clipping plane in meters, do not set non-zero
camera_far = 100                                               #far clipping plane in meters
object_location_desired = np.array([camera_width/2,camera_height/2])
                                                               #center the object to middle of image 


# Start the connection to the physics server
physicsClient = p.connect(p.GUI)#(p.DIRECT)
time_step = 0.001
p.resetSimulation()
p.setTimeStep(time_step)
p.setGravity(0, 0, -9.8)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.loadURDF('plane.urdf')

#reset debug gui camera position so we can see the robot up close
p.resetDebugVisualizerCamera( cameraDistance=1, cameraYaw=30, cameraPitch=-52, cameraTargetPosition=[0,0,.5])

''' Create Robot Instance'''
pandaUid = p.loadURDF(os.path.join(pybullet_data.getDataPath(), "franka_panda/panda.urdf"),useFixedBase=True)
p.resetBasePositionAndOrientation(pandaUid, [0, 0, 0], [0, 0, 0, 1])
initialJointPosition = [0,-np.pi/4,np.pi/4,-np.pi/4,np.pi/4,np.pi/4,np.pi/4,0,0,0,0,0]
robot = eye_in_hand_robot(pandaUid,initialJointPosition)
p.stepSimulation() # need to do this to initialize robot

# Place a green square plate
plate_size = 3
plate_thickness = 0.1
plate_position = np.array([0,2,plate_thickness/2])
geomPlate = p.createCollisionShape(p.GEOM_BOX, halfExtents=[plate_size/2, plate_size/2, plate_thickness/2])
visualPlate = p.createVisualShape(p.GEOM_BOX, halfExtents=[plate_size/2, plate_size/2, plate_thickness/2],
                                  rgbaColor=[0,1,0,1])
plateId = p.createMultiBody(baseMass=0,
                            baseCollisionShapeIndex=geomPlate,
                            baseVisualShapeIndex=visualPlate,
                            basePosition=plate_position,
                            baseOrientation=p.getQuaternionFromEuler([0,0,0]))
# Place a box
box_length, box_width, box_depth = 0.4, 0.4, 0.4
box_pos_rel = np.array([0.5, -0.5, box_depth/2+0.05])
target_box_center = plate_position + box_pos_rel
geomBox = p.createCollisionShape(p.GEOM_BOX, halfExtents=[box_length/2, box_width/2, box_depth/2])
visualBox = p.createVisualShape(p.GEOM_BOX, halfExtents=[box_length/2, box_width/2, box_depth/2],
                                rgbaColor=[1,0,0,1])
boxId = p.createMultiBody(baseMass=0,
                          baseCollisionShapeIndex=geomBox,
                          baseVisualShapeIndex=visualBox,
                          basePosition=target_box_center,
                          baseOrientation=p.getQuaternionFromEuler([0,0,0]))

# Place a blue sphere
sphere_radius = 0.25
sphere_pos_rel = np.array([-0.5, 0.5, sphere_radius+0.05])
sphere_position = plate_position + sphere_pos_rel
geomSphere = p.createCollisionShape(p.GEOM_SPHERE, radius=0.5*sphere_radius)
visualSphere = p.createVisualShape(p.GEOM_SPHERE, radius=sphere_radius, rgbaColor=[0,0,1,1])
sphereId = p.createMultiBody(baseMass=0,
                             baseCollisionShapeIndex=geomSphere,
                             baseVisualShapeIndex=visualSphere,
                             basePosition=sphere_position,
                             baseOrientation=p.getQuaternionFromEuler([0,0,0]))



# Text description of the object to find
object_type = input("Enter the object type (e.g., 'red cube', 'blue ball'): ")
dynamic = int(input("Enter 0 for static, 1 for dynamic:"))

target_description = f"The {object_type} on the green plane"
text_features = encode_text(target_description)

if object_type == "red cube":
    # Use the position of the box as the target object center
    object_center = target_box_center
    objId = boxId
elif object_type == "blue ball":
    # Use the position of the sphere as the target object center
    object_center = sphere_position
    objId = sphereId
else:
    print("Unknown object type, defaulting to box.")
    object_center = target_box_center
    objId = boxId



K_p_x = 2                                                    #Proportional control gain for translation
K_p_Omega = 2                                               #Proportional control gain for rotation  
motion_speed = 0.5
initial_vel_guess = np.array([0, 1]) * motion_speed
ff_gain = 100 #100 61; 0 67
log = False
initial_flag = True
error_ls = []
sample_error_ls = []
similarity_ls = []

for ITER in range(200):
    p.stepSimulation()
    if dynamic:
        object_pos=[object_center[0]+motion_speed*np.sin(np.pi/2+ITER/10), object_center[1]+motion_speed*np.sin(ITER/20), object_center[2]]
        p.resetBasePositionAndOrientation(objId,object_pos,p.getQuaternionFromEuler([0, 0, 0]))
    else:
        p.resetBasePositionAndOrientation(objId, object_center, p.getQuaternionFromEuler([0,0,0]))
    
    cameraPosition, cameraOrientation = robot.get_ee_position()
    rgb, depth = get_camera_img_float(cameraPosition, cameraOrientation)
    pil_img = Image.fromarray(rgb)

    # Generate candidate bounding boxes (for demonstration, assume we have a function)
    candidate_boxes = generate_candidate_boxes(rgb)

    best_similarity = -1
    best_box = None

    # Preprocess all candidate boxes first
    crops = []
    for box in candidate_boxes:
        x1, y1, x2, y2 = box
        cropped_img = pil_img.crop((x1, y1, x2, y2))
        crop_tensor = preprocess(cropped_img).unsqueeze(0)  # shape: (1, C, H, W)
        crops.append(crop_tensor)

    # Stack all crops into a single batch
    batch = torch.cat(crops, dim=0).to(device)  # shape: (N, C, H, W) where N = number of candidate boxes

    # Run model once on entire batch
    with torch.no_grad():
        image_features = model.encode_image(batch)  # shape: (N, D)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

    # Compute similarities for all boxes at once
    similarities = (image_features @ text_features.T).squeeze()  # shape: (N,)

    # Find best matching box
    best_idx = similarities.argmax().item()
    best_box = candidate_boxes[best_idx]
    best_similarity = similarities[best_idx].item()
    similarity_ls.append(best_similarity)
    # Now best_box corresponds to the region where CLIP thinks the described object is
    # Extract the center of this box as your target pixel location
    (x1, y1, x2, y2) = best_box
    u_px = (x1 + x2) // 2
    v_px = (y1 + y2) // 2
    object_loc=np.array([u_px, v_px])


    if initial_flag:
        # We are very confident in observation model, but not so much in motion model since we are not sure about the velocity
        obj_filter = uv_filter(object_loc[0], object_loc[1], initial_vel_guess, np.diag([1, 1, 1000, 1000]), np.diag([ 10, 10, 100, 100 ]), np.diag([1e-6, 1e-6]))
        initial_flag = False
    else:
        obj_filter.predict(time_step)
        obj_filter.update(object_loc)
    ''' Do some things here to get robot control'''
    imageJacobian = getImageJacobian(object_loc[0], object_loc[1], depth, camera_focal_depth, camera_width, camera_height)
    # Instead of using current location, we use the predicted location of next frame as desired location
    delta_X, delta_Omega = findCameraControl(object_location_desired, object_loc+(ff_gain*time_step*obj_filter.state[2:]).astype(int), imageJacobian, K_p_x, K_p_Omega)
    error_ls.append(np.linalg.norm(object_loc - object_location_desired))
    print((ff_gain*time_step*obj_filter.state[2:]).astype(int))
    delta_X_world = cameraOrientation @ delta_X
    delta_Omega_world = cameraOrientation @ delta_Omega
    J = robot.get_jacobian_at_current_position() 
    dq = 0.003*np.linalg.pinv(J) @ np.concatenate((delta_X_world, delta_Omega_world))
    current_q = robot.get_current_joint_angles()

    new_jointPositions = (current_q + dq)#.tolist()
    robot.set_joint_position(new_jointPositions)
    
    # show image
    cv2.imshow("depth", depth)
    cv2.imshow("rgb", rgb)
    cv2.waitKey(1)
    
    # Show debug info
    debug_img = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
    cv2.circle(debug_img, (u_px, v_px), 5, (0,0,0), 2)
    cv2.circle(debug_img, (int(object_location_desired[0]), int(object_location_desired[1])), 5, (255,0,0), 2)
    cv2.rectangle(debug_img, (x1, y1), (x2, y2), (0, 0, 0), 2)  # Red rectangle
    
    cv2.imshow("Camera RGB", debug_img)



    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    


#close the physics server
cv2.destroyAllWindows()    
p.disconnect() 
error_ls = np.array(error_ls)
similarity_ls = np.array(similarity_ls)
sample_error_ls = np.array(sample_error_ls)
print('Mean Error:', np.mean(error_ls))



