
import fiona

# ~ gdb_filename = '/mnt/c/Users/Gabriel/Geo_data/Original_SPOT_images.gdb.zip'
gdb_filename = '/mnt/c/Users/Gabriel/Geo_data/Original_SPOT_images.gdb'


print(fiona.listlayers(gdb_filename))

with fiona.open(gdb_filename,'r') as gdb_open :
    
    print(len(gdb_open))
    # ~ for a in gdb_open:
        # ~ print(a)
