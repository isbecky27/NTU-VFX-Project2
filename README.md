# NTU-VFX-Project2

### Team Member
* R10922022 曾筱晴
* B09902028 曾翊綺

### Environment
* python == 3.9.6
* numpy == 1.20.1
* opencv-python == 4.5.3.56

### Directory Structure
```
hw2_[13]
 ├─ code 
 ├─ data  
 │   ├─ grail
 │   └─ parrington                            
 └─ result        
     ├─ grail
     └─ parrington         
```

### Run Code
```python=
python main.py [--data_path DATA_PATH] [--result_path RESULT_PATH]
               [--series_of_images SERIES_OF_IMAGES] [--focal_length_filename FOCAL_LENGTH_FILENAME]
```
* Optional arguments 
    * `--data_path` : Path to the directory that contains series of images.
    * `--result_path` : Path to the directory that stores all of results.
    * `--series_of_images` : The folder of a series of images that contains images and focal length file.
    * `--focal_length_filename` : The name of the file where focal length information is stored.
    
* Display the usage message of all arguments
```python=
python main.py --help
```
