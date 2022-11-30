import os

out_filename = 'all_files.txt'


# base_path = '/mnt/c/GeoData/New_scenes/'
# base_path = '/mnt/c/GeoData/New_scenes/Scenes_BW_New/'
# base_path = '/mnt/c/GeoData/New_scenes/Scenes_Color_BW_Duplicates/'
base_path = '/mnt/c/GeoData/New_scenes/Scenes_Color_New/'

Do_rename = True
# Do_rename = False


# min size in bytes
min_size = 100 * 1024 * 1024
# min_size = 10 * 1024 * 1024

# max_size = 101 * 1024 * 1024
max_size = 10 * 1024 * 1024 * 1024

accepted_ext = [
    '.TIF',
    '.tif',
    '.Tif',
]
# 
# accepted_ext = [
#     '.TIFF',
#     '.tiff',
#     '.Tiff',
# ]

all_found_files = []


for root, dirs, files in os.walk(base_path):
   for filename in files:
        full_filename = os.path.join(root, filename)

        base,ext =  os.path.splitext(full_filename)

        if (ext in accepted_ext) and (os.path.getsize(full_filename) > min_size) and (os.path.getsize(full_filename) < max_size):

            all_found_files.append(full_filename)


# print(all_found_files)

if Do_rename:

    all_bases = []
    all_n_duplicates = []
    is_duplicate = []

    for full_filename in all_found_files:
            
        base,ext =  os.path.splitext(full_filename)
        base = os.path.basename(base)

        n_duplicate = 0
        for i in range(len(all_bases)):
            if base == all_bases[i] :
                n_duplicate += 1
                is_duplicate[i] = True

        all_bases.append(base)
        all_n_duplicates.append(n_duplicate)
        is_duplicate.append(n_duplicate > 0)
# 
        # print(full_filename)
        # print(int(os.path.getsize(full_filename)/(1024*1024)))
#         print(n_duplicate)
        # print('')
# 
    for i in range(len(all_found_files)):
        
        full_filename = all_found_files[i]
        n_duplicate = all_n_duplicates[i]

        if (is_duplicate[i]):

            base,ext =  os.path.splitext(full_filename)

            new_file = base+'_'+str(n_duplicate+1).zfill(3)+ext
        # 
            print(full_filename)
            print(new_file)
            print('')

            # os.rename(full_filename, new_file)



with open(out_filename,'w',encoding = 'utf-8') as f:

    for full_filename in all_found_files:
            
        f.write("'")
        f.write(full_filename)
        f.write("',\n")