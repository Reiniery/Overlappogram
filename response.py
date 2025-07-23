from __future__ import annotations

import numpy as np
from ndcube import NDCube
import matplotlib.pyplot as plt
__all__ = ["prepare_response_function"]


def prepare_response_function(
    response_cube: NDCube, field_angle_range=None, response_dependency_list=None, fov_width=2
) -> (np.ndarray, float, float):
   
    
    if response_cube.data.ndim ==4:
        num_dep, num_field_angles, A, rsp_func_width = np.shape(response_cube.data)
    else: 
        # from Dyana Beabout
        num_dep, num_field_angles, rsp_func_width = np.shape(response_cube.data)

    dependency_list = [t for (_, t) in response_cube.meta["temperatures"]]
    dependency_list = np.round(dependency_list, decimals=1)
    field_angle_list = [a for (_, a) in response_cube.meta["field_angles"]]
    field_angle_list = np.round(field_angle_list, decimals=2)

    if response_dependency_list is None:
        dep_index_list = [i for (i, _) in response_cube.meta["temperatures"]]
        dep_list_deltas = abs(np.diff(dependency_list))
        max_dep_list_delta = max(dep_list_deltas)
    else:
        dep_list_deltas = abs(np.diff(dependency_list))
        max_dep_list_delta = max(dep_list_deltas)
        dep_index_list = []
        for dep in response_dependency_list:
            delta_dep_list = abs(dependency_list - dep)
            dep_index = np.argmin(delta_dep_list)
            if abs(dependency_list[dep_index] - dep) < max_dep_list_delta:
                dep_index_list = np.append(dep_index_list, dep_index)
        new_index_list = [*set(dep_index_list)]
        new_index_list = np.array(new_index_list, dtype=np.int32)
        new_index_list.sort()
        dep_index_list = new_index_list
        dependency_list = dependency_list[new_index_list]

    num_deps = len(dependency_list)

    field_angle_list_deltas = abs(np.diff(field_angle_list))
    max_field_angle_list_delta = max(field_angle_list_deltas)
    if field_angle_range is None:
        begin_slit_index = np.int64(0)
        end_slit_index = np.int64(len(field_angle_list) - 1)
        field_angle_range_index_list = [begin_slit_index, end_slit_index]
        field_angle_range_list = field_angle_list[field_angle_range_index_list]
    else:
        angle_index_list = []
        for angle in field_angle_range:
            delta_angle_list = abs(field_angle_list - angle)
            angle_index = np.argmin(delta_angle_list)
            if abs(field_angle_list[angle_index] - angle) < max_field_angle_list_delta:
                angle_index_list = np.append(angle_index_list, angle_index)
        new_index_list = [*set(angle_index_list)]
        new_index_list = np.array(new_index_list, dtype=np.int32)
        new_index_list.sort()
        field_angle_range_index_list = new_index_list
        field_angle_range_list = field_angle_list[new_index_list]
        begin_slit_index = field_angle_range_index_list[0]
        end_slit_index = field_angle_range_index_list[1]
        num_field_angles = (end_slit_index - begin_slit_index) + 1

    # Check if number of field angles is even.
    calc_half_fields_angles = divmod(num_field_angles, 2)
    if calc_half_fields_angles[1] == 0.0:
        end_slit_index = end_slit_index - 1
        field_angle_range_index_list[1] = end_slit_index
        field_angle_range_list[1] = field_angle_list[end_slit_index]
        num_field_angles = (end_slit_index - begin_slit_index) + 1

    calc_num_slits = divmod(num_field_angles, fov_width)
    num_slits = int(calc_num_slits[0])
    # Check if number of slits is even.
    calc_half_num_slits = divmod(num_slits, 2)
    if calc_half_num_slits[1] == 0.0:
        num_slits -= 1
    half_slits = divmod(num_slits, 2)

    half_fov = divmod(fov_width, 2)

    center_slit = divmod(end_slit_index - begin_slit_index, 2) + begin_slit_index

    begin_slit_index = center_slit[0] - half_fov[0] - (half_slits[0] * fov_width)
    end_slit_index = center_slit[0] + half_fov[0] + (half_slits[0] * fov_width)

    num_field_angles = (end_slit_index - begin_slit_index) + 1
    field_angle_range_index_list = [begin_slit_index, end_slit_index]
    field_angle_range_list = field_angle_list[field_angle_range_index_list]

    response_count = 0

    # Low Fip and High Fip Changes - Rei
    if response_cube.data.ndim==4:
        low_fip_response=response_cube.data[:,:,0,:]
        high_fip_response=response_cube.data[:,:,1,:]

        response_function = np.zeros((num_deps * num_slits * A  , rsp_func_width), dtype=np.float32) #(T*F*A, E)
        
        for dep_idx, index in enumerate(dep_index_list):
            slit_count = 0
            for slit_idx,slit_num in enumerate(range(
                    center_slit[0] - (half_slits[0] * fov_width),
                    center_slit[0] + ((half_slits[0] * fov_width) + 1),
                    fov_width,
                )): 
                    num_select_slits= len(range(center_slit[0] - (half_slits[0] * fov_width),
            center_slit[0] + ((half_slits[0] * fov_width) + 1),
            fov_width))
                    out_idx=dep_idx * num_select_slits +slit_idx
                    out_idx_high = out_idx +num_deps * num_select_slits
                    
                    if fov_width == 1:
                        response_function[(num_deps * slit_count) + response_count, :] = response_cube.data[index, slit_num, :] #need to change
                    else:
                        # Check if even FOV.
                        if half_fov[1] == 0: #need to change
                            response_function[(num_deps * slit_count) + response_count, :] = (
                                response_cube.data[
                                    index,
                                    slit_num - (half_fov[0] - 1) : slit_num + (half_fov[0] - 1) + 1,
                                    :,
                                ].mean(axis=0) # Changed to Mean instead of Sum
                                #.mean(axis=1)
                                + (response_cube.data[index, slit_num - half_fov[0], :] * 0.5)
                                + (response_cube.data[index, slit_num + half_fov[0], :] * 0.5)
                            )
                        else:
                            response_function[out_idx, :] = low_fip_response[
                                index,
                                slit_num - half_fov[0] : slit_num + half_fov[0] + 1,
                                :,
                            ].mean(axis=0) #.sum)(axis=0)
                            response_function[out_idx_high , :] = high_fip_response[
                                index,
                                slit_num - half_fov[0] : slit_num + half_fov[0] + 1,
                                :,
                            ].mean(axis=0) #.sum)(axis=0)
                            
                        slit_count += 1
                    response_count += 1
    else:
        response_function = np.zeros((num_deps * num_slits, rsp_func_width), dtype=np.float32)
        for index in dep_index_list:
            # Smooth over dependence.
            slit_count = 0
            for slit_num in range(
                center_slit[0] - (half_slits[0] * fov_width),
                center_slit[0] + ((half_slits[0] * fov_width) + 1),
                fov_width,
            ):  
                if fov_width == 1:
                    response_function[(num_deps * slit_count) + response_count, :] = response_cube.data[index, slit_num, :]
                else:
                    # Check if even FOV.
                    if half_fov[1] == 0:
                        response_function[(num_deps * slit_count) + response_count, :] = (
                            response_cube.data[
                                index,
                                slit_num - (half_fov[0] - 1) : slit_num + (half_fov[0] - 1) + 1,
                                :,
                            ].mean(axis=0) # Changed to Mean instead of Sum
                            #.mean(axis=1)
                            + (response_cube.data[index, slit_num - half_fov[0], :] * 0.5)
                            + (response_cube.data[index, slit_num + half_fov[0], :] * 0.5)
                        )
                    else:
                        
                        response_function[(num_deps * slit_count) + response_count, :] = response_cube.data[
                            index,
                            slit_num - half_fov[0] : slit_num + half_fov[0] + 1,
                            :,
                        ].mean(axis=0) #.sum)(axis=0)
                        
                slit_count += 1
            response_count += 1
     
   
    return response_function.transpose(), num_slits, num_deps

#needs update for lowfip high fip
def prepare_emocci_filter(em_filter, dep_index_list, field_angle_list, field_angle_range, fov_width, lowfiphighfip=False):
    num_deps,rsp_func_width,num_field_angles= em_filter.shape
    num_bins = len(dep_index_list)
    half_fov = divmod(fov_width, 2)
    # Step 1: Get matching slit indices using same logic as response function
    angle_index_list = []
    for angle in field_angle_range:
        delta_angle_list = abs(field_angle_list - angle)
        angle_index = np.argmin(delta_angle_list)
        angle_index_list.append(angle_index)

    angle_index_list = sorted(set(angle_index_list))
    
    begin_slit_index, end_slit_index = angle_index_list[0], angle_index_list[1]
    num_field_angles = (end_slit_index - begin_slit_index) + 1
 

    # Step 2: Enforce odd number of field angles
    if num_field_angles % 2 == 0:
        end_slit_index -= 1
        num_field_angles = end_slit_index - begin_slit_index + 1

    # Step 3: Calculate slits
    num_slits = num_field_angles // fov_width
    if num_slits % 2 == 0:
        num_slits -= 1
    half_slits = divmod(num_slits ,2)
    center_slit = divmod(end_slit_index - begin_slit_index, 2) + begin_slit_index

    begin_slit_index = center_slit[0] - half_fov[0] -(half_slits[0] - fov_width)
    end_slit_index = center_slit[0] + half_fov[0] +(half_slits[0] - fov_width)

    em_mask=np.zeros((num_slits*num_bins*2,rsp_func_width)) #(T*F*A, e)

    response_count=0
    
    if lowfiphighfip:
        for dep_idx in range(len(em_filter)):
           

            slit_count=0
            bin_mask = em_filter[dep_idx] 
            for slit_idx,slit_num in enumerate(range(
            center_slit[0] - (half_slits[0] * fov_width),
                center_slit[0] + ((half_slits[0] * fov_width) + 1),
                fov_width,
            )):
                
                num_select_slits= len(range(center_slit[0] - (half_slits[0] * fov_width),
                center_slit[0] + ((half_slits[0] * fov_width) + 1),
                fov_width))
                out_idx=dep_idx * num_select_slits +slit_idx
                out_idx_high = out_idx +num_deps * num_select_slits    



                
                if fov_width == 1:
                    em_mask[(num_deps * slit_count) + response_count, :] = bin_mask[:,slit_num]
                else:
                    result = bin_mask[:, slit_num - half_fov[0]: slit_num + half_fov[0] ].mean(axis=1) #.sum(axis=1)
                    
                    em_mask[out_idx, :] = result
                    em_mask[out_idx_high]=result
            
                slit_count+=1
            
            response_count+=1




        
       

    else:
        em_mask=np.zeros((num_slits*num_bins,rsp_func_width),dtype=np.float32)
        response_count=0
        
        for dep_idx in range(len(em_filter)):
            slit_count=0
            bin_mask = em_filter[dep_idx] 
            for slit_num in range(
            center_slit[0] - (half_slits[0] * fov_width),
                center_slit[0] + ((half_slits[0] * fov_width) + 1),
                fov_width,
            ):
                if fov_width == 1:
                    em_mask[(num_deps * slit_count) + response_count, :] = bin_mask[:,slit_num]
                else:
                    result = bin_mask[:, slit_num - half_fov[0]: slit_num + half_fov[0] ].mean(axis=1) #.sum(axis=1)
                    em_mask[(num_deps * slit_count) + response_count, :] = result
            
                slit_count+=1
            
            response_count+=1
    
    plt.imshow(em_mask.T,origin='lower')
    plt.show()
    
    return em_mask.transpose(),num_slits, num_deps
