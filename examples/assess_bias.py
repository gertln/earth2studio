import numpy as np
from tqdm import tqdm
from collections import OrderedDict
from earth2studio.utils.coords import CoordSystem, map_coords, split_coords
from earth2studio.utils.time import to_time_array
import torch
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel
from physicsnemo.distributed import DistributedManager
DistributedManager.initialize()

device = 'cuda'
from earth2studio.data import DataSource, fetch_data
#from localh5 import LocalArchiveHDF5
from earth2studio.models.px import SFNO
package = SFNO.load_default_package()
px = SFNO.load_model(package)

px = px.to(device)
from earth2studio.data import ARCO
data = ARCO()
#data = LocalArchiveHDF5('/mnt/e/data/ERA5_h5/era5_75var/2010_.h5', '/mnt/e/data/ERA5_h5/era5_75var/data.json')

import h5py
with h5py.File('/mnt/e/data/ERA5_h5/r64x128_75v/2010_lr.h5', 'r') as f:
    x_era5 = f['fields'][:,4]
x_era5 = torch.tensor(x_era5,device=device)

model_ic = px.input_coords()

nsteps = 360

import pandas as pd
time = pd.date_range(start="2010-01-01",end="2010-12-31",freq="6h")
time = to_time_array(time)

#time = to_time_array(["2010-01-01"])
mbias = torch.empty((nsteps+1,1,64,128),dtype=torch.float,device=device)
gbias = torch.empty((nsteps+1,8760//6),dtype=torch.float,device=device)
n_samples = torch.zeros((nsteps+1,1),dtype=torch.float,device=device)
for i_time in range(len(time)):
    i_era5_offset = int(((time[i_time]-time[0])/np.timedelta64(1, 'h')))
    x, coords = fetch_data(
            source=data,
            time=time[[i_time]],
            variable=model_ic["variable"],
            lead_time=model_ic["lead_time"],
            device=device,
        )

    # # Set up IO backend
    # output_coords=OrderedDict({})
    # total_coords = px.output_coords(px.input_coords()).copy()
    # for key, value in px.output_coords(
    #     px.input_coords()
    # ).items():  # Scrub batch dims
    #     if value.shape == (0,):
    #         del total_coords[key]
    # total_coords["time"] = time
    # total_coords["lead_time"] = np.asarray(
    #     [
    #         px.output_coords(px.input_coords())["lead_time"] * i
    #         for i in range(nsteps + 1)
    #     ]
    # ).flatten()
    # total_coords.move_to_end("lead_time", last=False)
    # total_coords.move_to_end("time", last=False)

    # for key, value in total_coords.items():
    #     total_coords[key] = output_coords.get(key, value)
    # var_names = total_coords.pop("variable")

    # Map lat and lon if needed
    x, coords = map_coords(x, coords, px.input_coords())
    # Create prognostic iterator
    model = px.create_iterator(x, coords)

    
    with tqdm(total=nsteps + 1, desc="Running inference", position=1) as pbar:
        for step, (x, coords) in enumerate(model):
            # Subselect domain/variables as indicated in output_coords
            #x, coords = map_coords(x, coords, output_coords)
            
            xi = F.interpolate(x[0,:,[4]],size=(64,128),mode='area')
            bias = xi[0]-x_era5[i_era5_offset+step*6]
            gbias[step,i_era5_offset//6] = bias.mean()
            n_samples[step] += 1
            mbias[step] += (bias - mbias[step,0])/n_samples[step,0]
            pbar.update(1)
            if step == nsteps:
                break
    print(1)