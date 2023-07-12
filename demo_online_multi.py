import os
import cv2
import time
import joblib
import argparse
import warnings
import traceback
import numpy as np
from tqdm import tqdm

from PHALP import PHALP_tracker
from deep_sort_ import nn_matching
from deep_sort_.detection import Detection
from deep_sort_.tracker import Tracker

from utils.make_video import render_frame_main_online
from utils.utils import FrameExtractor, str2bool

warnings.filterwarnings('ignore')
  
        
def test_tracker(opt, phalp_tracker_multi):
    
    eval_keys       = ['tracked_ids', 'tracked_bbox', 'tid', 'bbox', 'tracked_time']
    history_keys    = ['appe', 'loca', 'pose', 'uv'] if opt.render else []
    prediction_keys = ['prediction_uv', 'prediction_pose', 'prediction_loca'] if opt.render else []
    extra_keys_1    = ['center', 'scale', 'size', 'img_path', 'img_name', 'mask_name', 'conf']
    extra_keys_2    = ['smpl', '3d_joints', 'camera', 'embedding']
    history_keys    = history_keys + extra_keys_1 + extra_keys_2
    visual_store_   = eval_keys + history_keys + prediction_keys
    tmp_keys_       = ['uv', 'prediction_uv', 'prediction_pose', 'prediction_loca']
    

    for view_index in range(opt.num_views):                                          
        if(not(opt.overwrite) and os.path.isfile('out/' + opt.storage_folder + f"/view{view_index}" + '/results/' + str(opt.video_seq) + '.pkl')): return 0
        print(opt.storage_folder + f"/view{view_index}" + '/results/' + str(opt.video_seq))
    
    try:
        os.makedirs('out/' + opt.storage_folder, exist_ok=True)  
        for view_index in range(opt.num_views):
            os.makedirs('out/' + opt.storage_folder + f"/view{view_index}", exist_ok=True)  
            os.makedirs('out/' + opt.storage_folder + f"/view{view_index}" + '/results', exist_ok=True)  
            os.makedirs('out/' + opt.storage_folder + f"/view{view_index}" + '/_TMP', exist_ok=True)  
    except: pass
    

    for phalp_tracker in phalp_tracker_multi:
        phalp_tracker.eval()
        phalp_tracker.HMAR.reset_nmr(opt.res)    
    

    ############ multi-view tracker set up ##############
    tracker_multi = []
    for view in range(opt.num_views):
        print(f"Initialized Tracker {view}\n")
        metric  = nn_matching.NearestNeighborDistanceMetric(opt, opt.hungarian_th, opt.past_lookback)
        tracker = Tracker(opt, metric, max_age=opt.max_age_track, n_init=opt.n_init, phalp_tracker=phalp_tracker_multi[view], dims=[4096, 4096, 99])  ## dimension of apperance, pose, and location. 
        tracker_multi.append(tracker)
    

    try: 
        ############ multi-view set up ##############
        if (opt.multi_view):
            print("Multi-view activated...")
            main_path_to_frames_multi = []
            list_of_frames_multi = []
            list_of_shots_multi = []
            for j in range(1,(opt.num_views+1)):
                ## directory should looks like: _DEMO/video/1 or 2/img
                main_path_to_frames = opt.base_path + '/' + opt.video_seq + f"/{j}" + opt.sample 
                main_path_to_frames_multi.append(main_path_to_frames)
                list_of_frames = np.sort([i for i in os.listdir(main_path_to_frames) if '.jpg' in i])
                list_of_frames_multi.append(list_of_frames)
                list_of_shots = phalp_tracker_multi[j-1].get_list_of_shots(main_path_to_frames, list_of_frames, j-1) ## list of frames denoting a shot changes
                list_of_shots_multi.append(list_of_shots)
        

        track_frames_multi = []
        final_visuals_dic_multi = []
        for _ in range(opt.num_views):
            tracked_frames          = []
            track_frames_multi.append(tracked_frames)
            final_visuals_dic       = {} ##TODO: writing the files
            final_visuals_dic_multi.append(final_visuals_dic)

        ## Loop-1: from frame 1 to frame T, every frame is denoted as "t_". 
        for t_, frame_name_view1 in enumerate(tqdm(list_of_frames_multi[0])):
            if(opt.verbose): 
                print('\n\n\nTime: ', opt.video_seq, frame_name_view1, t_, time.time()-time_ if t_>0 else 0 )
                time_ = time.time()
            
            ############ multi-view detection set up ##############
            image_frame_list = []
            for view_index in range(opt.num_views):
                image_frames = []
                main_path_to_frames = main_path_to_frames_multi[view_index]
                for frame_name in list_of_frames_multi[view_index]:
                    image_frame = cv2.imread(main_path_to_frames + '/' + frame_name)
                    image_frame_obj = ImageFrame(image_frame)
                    image_frames.append(image_frame_obj)
                image_frame_list.append(image_frames)
            

            ############ detection ##############
            detections_multi = [] ## contains detection for frame t_ for multi-view
            for view_index in range(opt.num_views):
                ## image_frame: the image_frame at frame t_ with view 1 and view 2. frame_name: 
                pred_bbox, pred_masks, pred_scores, mask_names, gt = phalp_tracker_multi[view_index].get_detections(image_frame_list[view_index][t_].image_frame, list_of_frames_multi[view_index][t_], t_, view_index) 
                detections_element = DetectionWrap(pred_bbox,pred_masks,pred_scores,mask_names,gt)
                detections_multi.append(detections_element)
                 

            ############ HMAR ##############
            detection_data_multi = [] ## contains detection data for frame t_ for multi-view
            for view_index in range(opt.num_views):
                detections = []
                main_path_to_frames = main_path_to_frames_multi[view_index]
                frame_name = list_of_frames_multi[view_index][t_]
                image_frame_element = image_frame_list[view_index][t_]
                detections_element = detections_multi[view_index]
                for bbox, mask, score, mask_name, gt_id in zip(detections_element.pred_bbox, detections_element.pred_masks, detections_element.pred_scores, detections_element.mask_names, detections_element.gt):
                    if bbox[2]-bbox[0]<50 or bbox[3]-bbox[1]<100: continue
                    ## return all the necessary feature (pose, appe, etc.) given the bounding box and masks. 
                    detection_data = phalp_tracker_multi[view_index].get_human_apl(image_frame_element.image_frame, mask, bbox, score, [main_path_to_frames, frame_name], mask_name, t_, image_frame_element.measurments, gt_id)
                    ##TODO: BUG, DETECTION same name with wrapper class
                    detections.append(Detection(detection_data))
                detection_data_multi.append(detections)

            video_file_multi = []
            ### Loop-2: from view 1 to view S, every view is denoted as view_index" 
            for view_index in range(opt.num_views):
                tracker = tracker_multi[view_index]

                ############ tracking ##############
                tracker.predict() ## simply increment every age and time step by 1

                detections = detection_data_multi[view_index]
                frame_name = list_of_frames_multi[view_index][t_]
                opt.shot = 1 if t_ in list_of_shots_multi[view_index] else 0
                tracker.update(detections, t_, frame_name, opt.shot,detection_data_multi,view_index) ## during update, it performs association (i.e. matching)

                ##TODO: this is another implementation (closer to MvMHAT version of implementation)
                # ### perform matching
                # ## loop every tracklets view v > view u
                # if view_index > 0:
                #     for tracker_diff_view in tracker_multi[view_index-1]:
                #         #tracker_diff_view.update_spatio(tracker,t_) ## perform spatio association between views.
                #         tracker_diff_view.update


                ##TODO: we want to rewrite the final_visuals_dic for multiple view at one loop, rewrite every video from 1-1, 1-2, 1-3 depending on the view-index.

                ############ record the results ##############
                final_visuals_dic = final_visuals_dic_multi[view_index]
                final_visuals_dic.setdefault(frame_name, {'time': t_, 'shot': opt.shot})
                if(opt.render): final_visuals_dic[frame_name]['frame'] = image_frame_element = image_frame_list[view_index][t_].image_frame
                for key_ in visual_store_: final_visuals_dic[frame_name][key_] = []
                
                tracked_frames = track_frames_multi[view_index]
                for tracks_ in tracker.tracks:
                    if(frame_name not in tracked_frames): tracked_frames.append(frame_name)
                    if(not(tracks_.is_confirmed())): continue ## if the current tracklets subjects is not confirmed, just continued. 
                    
                    track_id        = tracks_.track_id
                    track_data_hist = tracks_.track_data['history'][-1]
                    track_data_pred = tracks_.track_data['prediction']

                    ## map the id, bbox, and time into the final visuals image at each frame.
                    final_visuals_dic[frame_name]['tid'].append(track_id)
                    final_visuals_dic[frame_name]['bbox'].append(track_data_hist['bbox'])
                    final_visuals_dic[frame_name]['tracked_time'].append(tracks_.time_since_update)

                    for hkey_ in history_keys:     final_visuals_dic[frame_name][hkey_].append(track_data_hist[hkey_])
                    for pkey_ in prediction_keys:  final_visuals_dic[frame_name][pkey_].append(track_data_pred[pkey_.split('_')[1]][-1])

                    if(tracks_.time_since_update==0):
                        final_visuals_dic[frame_name]['tracked_ids'].append(track_id)
                        final_visuals_dic[frame_name]['tracked_bbox'].append(track_data_hist['bbox'])
                        
                        if(tracks_.hits==opt.n_init):
                            for pt in range(opt.n_init-1):
                                track_data_hist_ = tracks_.track_data['history'][-2-pt]
                                track_data_pred_ = tracks_.track_data['prediction']
                                frame_name_      = tracked_frames[-2-pt]
                                final_visuals_dic[frame_name_]['tid'].append(track_id)
                                final_visuals_dic[frame_name_]['bbox'].append(track_data_hist_['bbox'])
                                final_visuals_dic[frame_name_]['tracked_ids'].append(track_id)
                                final_visuals_dic[frame_name_]['tracked_bbox'].append(track_data_hist_['bbox'])
                                final_visuals_dic[frame_name_]['tracked_time'].append(0)

                                for hkey_ in history_keys:    final_visuals_dic[frame_name_][hkey_].append(track_data_hist_[hkey_])
                                for pkey_ in prediction_keys: final_visuals_dic[frame_name_][pkey_].append(track_data_pred_[pkey_.split('_')[1]][-1])

                
                ##TODO:BUG FIX (rendering the same based video for two frames)
                ############ save the video ##############
                list_of_frames = list_of_frames_multi[view_index] 
                list_of_shots = list_of_shots_multi[view_index]
                if(opt.render and t_>=opt.n_init):
                    d_ = opt.n_init+1 if(t_+1==len(list_of_frames)) else 1
                    for t__ in range(t_, t_+d_):
                        frame_key          = list_of_frames[t__-opt.n_init] ##frame key is the same, don't think this is the issue here
                        rendered_, f_size  = render_frame_main_online(opt, phalp_tracker_multi[view_index], frame_key, final_visuals_dic[frame_key], opt.track_dataset, track_id=-100)      
                        if(t__-opt.n_init in list_of_shots): cv2.rectangle(rendered_, (0,0), (f_size[0], f_size[1]), (0,0,255), 4)
                        if(t__-opt.n_init==0):
                            ## Multi-view writes into sub-folder
                            file_name      = 'out/' + opt.storage_folder + f"/view{view_index}" '/PHALP_' + str(opt.video_seq) + '_'+ str(opt.detection_type) + '.mp4'
                            video_file     = cv2.VideoWriter(file_name, cv2.VideoWriter_fourcc(*'mp4v'), 30, frameSize=f_size)
                            video_file_multi.append(video_file)
                        ##TODO: Index out of range, fix just write the latest one, problem is nothing in the list at frame 6.
                        video_file_multi[view_index].write(rendered_)
                        del final_visuals_dic[frame_key]['frame']
                        for tkey_ in tmp_keys_:  del final_visuals_dic[frame_key][tkey_] 



        ### Need a for loop
        ##TODO: need re-implementing
        for view_index in range(opt.num_views):
            joblib.dump(final_visuals_dic_multi[view_index], 'out/' + opt.storage_folder + f"/view{view_index}" + '/results/' + opt.track_dataset + "_" + str(opt.video_seq) + opt.post_fix  + '.pkl')
            tracker = tracker_multi[view_index]
            if(opt.use_gt): joblib.dump(tracker.tracked_cost, 'out/' + opt.storage_folder + f"/view{view_index}" + '/results/' + str(opt.video_seq) + '_' + str(opt.start_frame) + '_distance.pkl')
            if(opt.render): video_file_multi[view_index].release()
    
    except Exception as e: 
        print(e)
        print(traceback.format_exc())     

    return phalp_tracker_multi


class DetectionWrap():
    """
    Wrapper class to store information of detections
    """
    def __init__(self, pred_bbox, pred_masks, pred_scores, mask_names, gt):
        self.pred_bbox = pred_bbox
        self.pred_masks = pred_masks
        self.pred_scores = pred_scores
        self.mask_names = mask_names
        self.gt = gt

class ImageFrame():
    """
    Wrapper class to store information of indidividual image frame
    """
    def __init__(self, image_frame):
        self.image_frame = image_frame
        img_height, img_width, _  = image_frame.shape
        self.img_height = img_height
        self.img_width = img_width
        new_image_size = max(img_height, img_width)
        self.new_image_size = new_image_size
        self.top = (new_image_size - img_height)//2
        self.left = (new_image_size - img_width)//2
        self.measurments  = [img_height, img_width, new_image_size, self.left, self.top]

class options():
    def __init__(self):

        self.parser = argparse.ArgumentParser(description='PHALP_pixel Tracker')
        self.parser.add_argument('--batch_id', type=int, default='-1')
        self.parser.add_argument('--track_dataset', type=str, default='posetrack')
        self.parser.add_argument('--predict', type=str, default='APL')
        self.parser.add_argument('--storage_folder', type=str, default='Videos_v20.000')
        self.parser.add_argument('--distance_type', type=str, default='A5')
        self.parser.add_argument('--use_gt', type=str2bool, nargs='?', const=True, default=False)
        self.parser.add_argument('--overwrite', type=str2bool, nargs='?', const=True, default=False)

        self.parser.add_argument('--alpha', type=float, default=0.1)
        self.parser.add_argument('--low_th_c', type=float, default=0.95)
        self.parser.add_argument('--hungarian_th', type=float, default=100.0)
        self.parser.add_argument('--track_history', type=int, default=7)
        self.parser.add_argument('--max_age_track', type=int, default=20)
        self.parser.add_argument('--n_init',  type=int, default=1)
        self.parser.add_argument('--max_ids', type=int, default=50)
        self.parser.add_argument('--verbose', type=str2bool, nargs='?', const=True, default=False)
        self.parser.add_argument('--detect_shots', type=str2bool, nargs='?', const=True, default=False)
        
        self.parser.add_argument('--base_path', type=str)
        self.parser.add_argument('--video_seq', type=str, default='_DATA/posetrack/list_videos_val.npy')
        self.parser.add_argument('--youtube_id', type=str, default="xEH_5T9jMVU")
        self.parser.add_argument('--all_videos', type=str2bool, nargs='?', const=True, default=True)
        self.parser.add_argument('--store_mask', type=str2bool, nargs='?', const=True, default=True)

        self.parser.add_argument('--render', type=str2bool, nargs='?', const=True, default=False)
        self.parser.add_argument('--render_type', type=str, default='HUMAN_HEAD_FAST')
        self.parser.add_argument('--render_up_scale', type=int, default=2)
        self.parser.add_argument('--res', type=int, default=256)
        self.parser.add_argument('--downsample',  type=int, default=1)
        
        self.parser.add_argument('--encode_type', type=str, default='3c')
        self.parser.add_argument('--cva_type', type=str, default='least_square')
        self.parser.add_argument('--past_lookback', type=int, default=1)
        self.parser.add_argument('--mask_type', type=str, default='feat')
        self.parser.add_argument('--detection_type', type=str, default='mask2')
        self.parser.add_argument('--start_frame', type=int, default='1000')
        self.parser.add_argument('--end_frame', type=int, default='1100')
        self.parser.add_argument('--store_extra_info', type=str2bool, nargs='?', const=True, default=False)

        ### Multi-vie argument set-up
        self.parser.add_argument('--multi_view', type=str2bool, nargs='?', const=True, default=True)
        self.parser.add_argument('--num_views', type=int, default=2)
    
    def parse(self):
        self.opt          = self.parser.parse_args()
        self.opt.sample   = ''
        self.opt.post_fix = ''
        return self.opt
   

if __name__ == '__main__':
    
    opt                   = options().parse()

    phalp_tracker_multi   =   []


    view_num              = 2
    for i in range(view_num):
        phalp_tracker         = PHALP_tracker(opt)
        phalp_tracker.cuda()
        phalp_tracker.eval()
        phalp_tracker_multi.append(phalp_tracker)

    if(opt.track_dataset=='test'):   
        video    = "multi_view_test1"

        #os.system("rm -rf " + "_DEMO/" + video)
        #os.makedirs("_DEMO/" + video, exist_ok=True)    
        #os.makedirs("_DEMO/" + video + "/img", exist_ok=True)    


        #fe = FrameExtractor("_DEMO/" + video + f"/{video}.mp4")
        #print('Number of frames: ', fe.n_frames)
        #fe.extract_frames(every_x_frame=1, img_name='', dest_path= "_DEMO/" + video + "/img/", start_frame=1100, end_frame=1300)

        opt.base_path       = '_DEMO/'
        opt.video_seq       = video
        opt.sample          =  '/img/'
        test_tracker(opt, phalp_tracker_multi)


            