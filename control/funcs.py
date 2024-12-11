import numpy as np
import pybullet as p
import matplotlib.pyplot as plt
from PIL import Image
import torch
from scipy.spatial.transform import Rotation as Rot            #can use this to apply angular rotations to coordinate frames
#camera (don't change these settings)
camera_width = 512                                             #image width
camera_height = 512                                            #image height
camera_fov = 120                                                #field of view of camera
camera_focal_depth = 0.5*camera_height/np.tan(0.5*np.pi/180*camera_fov) 
                                                               #focal depth in pixel space
camera_aspect = camera_width/camera_height                     #aspect ratio
camera_near = 0.02                                             #near clipping plane in meters, do not set non-zero
camera_far = 100                                               #far clipping plane in meters
def draw_coordinate_frame(position, orientation, length, frameId = []):
    '''
    Draws a coordinate frame x,y,z with scaled lengths on the axes 
    in a position and orientation relative to the world coordinate frame
    pos: 3-element numpy array
    orientation: 3x3 numpy matrix
    length: length of the plotted x,y,z axes
    frameId: a unique ID for the frame. If this supplied, then it will erase the previous location of the frame
    
    returns the frameId
    '''
    if len(frameId)!=0:
        p.removeUserDebugItem(frameId[0])
        p.removeUserDebugItem(frameId[1])
        p.removeUserDebugItem(frameId[2])
    
    lineIdx=p.addUserDebugLine(position, position + np.dot(orientation, [length, 0, 0]), [1, 0, 0])  # x-axis in red
    lineIdy=p.addUserDebugLine(position, position + np.dot(orientation, [0, length, 0]), [0, 1, 0])  # y-axis in green
    lineIdz=p.addUserDebugLine(position, position + np.dot(orientation, [0, 0, length]), [0, 0, 1])  # z-axis in blue

    return lineIdx,lineIdy,lineIdz

def opengl_plot_world_to_pixelspace(pt_in_3D_to_project, viewMat, projMat, imgWidth, imgHeight):
    ''' Plots a x,y,z location in the world in an openCV image
    This is used for debugging, e.g. given a known location in the world, verify it appears in the camera
    when using p.getCameraImage(...). The output [u,v], when plot with opencv, should line up with object 
    in the image from p.getCameraImage(...)
    '''
    pt_in_3D_to_project = np.append(pt_in_3D_to_project,1)
    #print('Point in 3D to project:', pt_in_3D_to_project)

    pt_in_3D_in_camera_frame = viewMat @ pt_in_3D_to_project
    #print('Point in camera space: ', pt_in_3D_in_camera_frame)

    # Convert coordinates to get normalized device coordinates (before rescale)
    uvzw = projMat @ pt_in_3D_in_camera_frame
    #print('after projection: ', uvzw)

    # scale to get the normalized device coordinates
    uvzw_NDC = uvzw/uvzw[3]
    #print('after normalization: ', uvzw_NDC)

    #x,y specifies lower left corner of viewport rectangle, in pixels. initial value is (0,)
    u = ((uvzw_NDC[0] + 1) / 2.0) * imgWidth
    v = ((1-uvzw_NDC[1]) / 2.0) * imgHeight

    return [int(u),int(v)]

    
def get_camera_view_and_projection_opencv(cameraPos, camereaOrn):
    '''Gets the view and projection matrix for a camera at position (3) and orientation (3x3)'''
    __camera_view_matrix_opengl = p.computeViewMatrix(cameraEyePosition=cameraPos,
                                                   cameraTargetPosition=cameraPos+camereaOrn[:,2],
                                                   cameraUpVector=-camereaOrn[:,1])

    __camera_projection_matrix_opengl = p.computeProjectionMatrixFOV(camera_fov, camera_aspect, camera_near, camera_far)        
    _, _, rgbImg, depthImg, _ = p.getCameraImage(camera_width, 
                                                 camera_height, 
                                                 __camera_view_matrix_opengl,
                                                 __camera_projection_matrix_opengl, 
                                                 renderer=p.ER_BULLET_HARDWARE_OPENGL)

    #returns camera view and projection matrices in a form that fits openCV
    viewMat = np.array(__camera_view_matrix_opengl).reshape(4,4).T
    projMat = np.array(__camera_projection_matrix_opengl).reshape(4,4).T
    return viewMat, projMat

def get_camera_img_float(cameraPos, camereaOrn):
    ''' Gets the image and depth map from a camera at a position cameraPos (3) and cameraOrn (3x3) in space. '''
    __camera_view_matrix_opengl = p.computeViewMatrix(cameraEyePosition=cameraPos,
                                                   cameraTargetPosition=cameraPos+camereaOrn[:,2],
                                                   cameraUpVector=-camereaOrn[:,1])

    __camera_projection_matrix_opengl = p.computeProjectionMatrixFOV(camera_fov, camera_aspect, camera_near, camera_far)        
    width, height, rgbImg, nonlinDepthImg, _ = p.getCameraImage(camera_width, 
                                                 camera_height, 
                                                 __camera_view_matrix_opengl,
                                                 __camera_projection_matrix_opengl, 
                                                 renderer=p.ER_BULLET_HARDWARE_OPENGL)

    #adjust for clipping and nonlinear distance i.e., 1/d (0 is closest, i.e., near, 1 is furthest away, i.e., far
    depthImgLinearized =camera_far*camera_near/(camera_far+camera_near-(camera_far-camera_near)*nonlinDepthImg)

    #convert to numpy and a rgb-d image
    rgb_image = np.array(rgbImg[:,:,:3], dtype=np.uint8)
    depth_image = np.array(depthImgLinearized, dtype=np.float32)
    return rgb_image, depth_image

def generate_candidate_boxes(img, num_boxes_horizontal=8, num_boxes_vertical=8):
    """
    Generate candidate bounding boxes by dividing the image into a grid.
    Each cell in the grid is considered a candidate bounding box.

    Parameters:
        img (np.array): The image as a numpy array (H, W, 3).
        num_boxes_horizontal (int): Number of candidate boxes along the width.
        num_boxes_vertical (int): Number of candidate boxes along the height.

    Returns:
        List[Tuple]: A list of candidate boxes in the form (x1, y1, x2, y2).
    """
    h, w, _ = img.shape
    box_width = w // num_boxes_horizontal
    box_height = h // num_boxes_vertical

    boxes = []
    for i in range(num_boxes_vertical):
        for j in range(num_boxes_horizontal):
            x1 = j * box_width
            y1 = i * box_height
            x2 = x1 + box_width
            y2 = y1 + box_height
            boxes.append((x1, y1, x2, y2))
    return boxes

def get_best_box(rgb, candidate_boxes, text_features, model, preprocess, device):
    # Extract all candidate crops
    crops = []
    for box in candidate_boxes:
        (x1, y1, x2, y2) = box
        pil_crop = Image.fromarray(rgb[y1:y2, x1:x2])
        crops.append(preprocess(pil_crop).unsqueeze(0))
    
    # Stack all crops into a single batch tensor
    batch = torch.cat(crops, dim=0).to(device)

    # Run the model once on the whole batch
    with torch.no_grad():
        image_features = model.encode_image(batch)
        image_features /= image_features.norm(dim=-1, keepdim=True)

    # Compute similarities
    similarities = (image_features @ text_features.T).squeeze()

    # Find best matching box
    best_idx = similarities.argmax().item()
    best_box = candidate_boxes[best_idx]
    best_similarity = similarities[best_idx].item()
    return best_box, best_similarity

def getImageJacobian(u_px,v_px,depthImg,focal_length, imgWidth, imgHeight):
    ''' Inputs: 
    u_px, v_px is pixel coordinates from image with top-left corner as 0,0
    depthImg is the depth map
    f is the focal length
    Outputs: image_jacobian, a 2x6 matrix'''
    z = depthImg[v_px,u_px]
    u = (u_px - imgWidth/2)
    v = (v_px - imgHeight/2)
    L = np.array([[-focal_length/z, 0, u/z, u*v/focal_length, -(focal_length**2 + u**2)/focal_length, v],
                  [0, -focal_length/z, v/z, (focal_length**2 + v**2)/focal_length, -u*v/focal_length, -u]])
    return L
    
def findCameraControl(object_loc_des, object_loc, image_jacobian, K_p_x, K_p_Omega):
    ''' Inputs:
    object_loc_des: desired [x,y] pixel locations for object
    object_loc: current [x,y] pixel locations as found from computer vision
    image_jacobian: the image jacobian 
    Outputs:
    delta_X: the scaled displacement in position of camera (world frame) to reduce the error
    delta_Omega: the scaled angular velocity of camera (world frame omega-x,y,z) to reduce the error
    '''
    #error in pixel space
    # K_p_x, K_p_Omega = 1, 1
    error = object_loc_des - object_loc
    #control law
    J_inv = np.linalg.pinv(image_jacobian)
    delta = J_inv @ error
    delta_X = K_p_x * delta[:3]
    delta_Omega = K_p_Omega * delta[3:]
    return delta_X, delta_Omega

def plot_final_results(similarity):
    """Create final summary plots of roll angle and tracking error"""
    plt.figure(figsize=(12, 8))
    
    # Roll angle subplot
    
    plt.plot(similarity, label='Similarity Change')
    plt.title('Similarity Change')
    plt.ylabel('Similarity')
    plt.xlabel('Time Step')
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    plt.show()