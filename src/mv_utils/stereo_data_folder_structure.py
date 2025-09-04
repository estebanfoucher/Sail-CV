import os
import json
import yaml
from typing import List

class StereoDataFolderStructure:
    """
    Structure of the stereo data folder:
    - stereo_data_folder
        - scene_<number>
            - camera_1
                - video.mp4
            - camera_2
                - video.mp4
            - config.yml
            - calibration_extrinsics_pattern_specs.yml
        - calibration_intrinsics_<number>
            - camera_1
                - video.mp4
            - camera_2
                - video.mp4

    """
    
    def __init__(self, folder_path, admitted_folder_names:List[str] = None):
        
        self.folder_path = folder_path
        self.admitted_folder_names = admitted_folder_names
        self.subfolders = self.get_subfolders()
        self.check_folder_structure(name=self.folder_path)
    
    def get_subfolders(self):
        if self.admitted_folder_names is None:
            return [f for f in os.listdir(self.folder_path) if f not in [".DS_Store"]]
        return [f for f in os.listdir(self.folder_path) if f in self.admitted_folder_names]
    
    def get_calibration_intrinsics_folders(self):
        return [f for f in self.subfolders if f.startswith("calibration_intrinsics")]
    
    def get_scene_folders(self):
        return [f for f in self.subfolders if f.startswith("scene")]
    
    def _check_folder_exists(self):
        if not os.path.exists(self.folder_path):
            raise ValueError(f"Folder {self.folder_path} does not exist")

    def _check_subfolders_names(self):
        admitted_prefixes = ["scene", "calibration_intrinsics"]
        ignored = [".DS_Store"]
        
        for subfolder in self.subfolders:
            if subfolder in ignored:
                continue
            if not any(subfolder.startswith(prefix) for prefix in admitted_prefixes):
                raise ValueError(f"Subfolder {subfolder} has an invalid name")
    
    def _check_scene_folder_structure(self, folder_path):
        """
        Check that the folder has two subfolders named camera_1 and camera_2 each with a video.mp4 file
        Check that the folder has a parameters.yml file
        Check that the folder has a extrinsics_calibration_pattern_specs.yml file
        """        
        expected_files = ["parameters.yml", "extrinsics_calibration_pattern_specs.yml"]
        expected_folders = ["camera_1", "camera_2"]
        
        actual_files = [f for f in os.listdir(folder_path) if f not in [".DS_Store"]]
        actual_folders = [f for f in os.listdir(folder_path) if f not in [".DS_Store"]]
        
        for expected in expected_files:
            if expected not in actual_files:
                raise ValueError(f"Expected '{expected}' in {folder_path}, but found: {actual_files}")
        
        for expected in expected_folders:
            if expected not in actual_folders:
                raise ValueError(f"Expected '{expected}' in {folder_path}, but found: {actual_folders}")
        
        # check that the camera_1 and camera_2 folders have a video.mp4 file
        assert self._check_folder_has_one_video_file(os.path.join(folder_path, "camera_1"))
        assert self._check_folder_has_one_video_file(os.path.join(folder_path, "camera_2"))
        
        # check that the parameters.yml file exists
        assert os.path.isfile(os.path.join(folder_path, "parameters.yml"))
        
        # check that the extrinsics_calibration_pattern_specs.yml file exists
        assert os.path.isfile(os.path.join(folder_path, "extrinsics_calibration_pattern_specs.yml"))
        
    def _check_calibration_intrinsics_folder_structure(self, folder_path):
        """
        Check that the folder has two subfolders named camera_1 and camera_2 each with a video.mp4 file
        """
        
        expected_folders = ["camera_1", "camera_2"]
        
        actual_folders = [f for f in os.listdir(folder_path) if f not in [".DS_Store"]]
        
        for expected in expected_folders:
            if expected not in actual_folders:
                raise ValueError(f"Expected '{expected}' in {folder_path}, but found: {actual_folders}")
        
        # check that the camera_1 and camera_2 folders have a video.mp4 file
        assert self._check_folder_has_one_video_file(os.path.join(folder_path, "camera_1"))
        assert self._check_folder_has_one_video_file(os.path.join(folder_path, "camera_2"))
        
        # assert each camera folder has a checkerboard_specs.yml file
        assert os.path.isfile(os.path.join(folder_path, "camera_1", "checkerboard_specs.yml")), f"checkerboard specs file not found in {os.path.join(folder_path, 'camera_1')}"
        assert os.path.isfile(os.path.join(folder_path, "camera_2", "checkerboard_specs.yml")), f"checkerboard specs file not found in {os.path.join(folder_path, 'camera_2')}"
    
    def _check_folder_has_one_video_file(self, folder_path):
        video_files = [f for f in os.listdir(folder_path) if f.endswith(".mp4") or f.endswith(".MP4")]
        if len(video_files) != 1:
            raise ValueError(f"Folder {folder_path} has {len(video_files)} video files, expected 1")
        return video_files[0]

    def check_folder_structure(self, name: str = None):
        self._check_folder_exists()
        self._check_subfolders_names()
        
        for subfolder in self.subfolders:
            if subfolder.startswith("scene"):
                self._check_scene_folder_structure(os.path.join(self.folder_path, subfolder))
            elif subfolder.startswith("calibration_intrinsics"):
                self._check_calibration_intrinsics_folder_structure(os.path.join(self.folder_path, subfolder))
            else:
                raise ValueError(f"Subfolder {subfolder} has an invalid name")
        print(f"{name} folder structure valid")

class SceneFolderStructure(StereoDataFolderStructure):
    '''
    It is also a stereo data folder structure, it has a scene_name attribute and will ignore all other scenes and non related calibration folders
    '''
    def __init__(self, folder_path, scene_name):
        assert scene_name is not None, "Scene name is required"
        super().__init__(folder_path)
        self.scene_name = scene_name
        self.admitted_folder_names = [scene_name, self.get_calibration_intrinsics_folder_name()]
        
    def get_calibration_intrinsics_folder_name(self):
        with open(os.path.join(self.folder_path, self.scene_name, "parameters.yml"), "r") as f:
            parameters = yaml.safe_load(f)
        return parameters["calibration_intrinsics_folder_name"]
           
def load_stereo_data_folder_structure() -> StereoDataFolderStructure:
    """
    Load the stereo data folder structure from the config.json file.
    """
    with open("config.json", "r") as f:
        config = json.load(f)
    STEREO_DATA_FOLDER_NAME = config["stereo_data_folder_name"]
    stereo_data_folder_structure = StereoDataFolderStructure(STEREO_DATA_FOLDER_NAME)
    return stereo_data_folder_structure

def load_scene_folder_structure(scene_name: str = None) -> SceneFolderStructure:
    """
    Load the scene folder structure from the config.json file.
    """
    with open("config.json", "r") as f:
        config = json.load(f)
    STEREO_DATA_FOLDER_NAME = config["stereo_data_folder_name"]
    scene_folder_structure = SceneFolderStructure(STEREO_DATA_FOLDER_NAME, scene_name)
    return scene_folder_structure

if __name__ == "__main__":
    with open("config.json", "r") as f:
        scene_name = json.load(f)["scene_name"]
    scene_folder_structure = load_scene_folder_structure(scene_name)