import bpy
import bmesh
from mathutils import Vector
from math import pi
import colorsys

import time
from random import random

import site
site.addsitedir(r'C:\Users\user\anaconda3\envs\yolo3\Lib\site-packages')

import pandas as pd
import numpy as np

D = bpy.data
O = D.objects
C = D.collections

CONFIGS_FILENAME = r'C:\Users\user\Documents\Blender\data\configs.csv'
RGBA_DICT = {
    'orange': (0.71, 0.22, 0.01, 1)
}

def trigger_update():
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.object.mode_set(mode='OBJECT')

def print_console(data):
    for window in bpy.context.window_manager.windows:
        screen = window.screen
        for area in screen.areas:
            if area.type == 'CONSOLE':
                override = {'window': window, 'screen': screen, 'area': area}
                bpy.ops.console.scrollback_append(override, text=str(data), type="OUTPUT") 

def delete_all_objects_from_collection(col_name, total_delete=False):
    assert col_name in D.collections.keys(), f'Collection \"{col_name}\" doesn\'t exist'
    to_delete = C[col_name].objects

    if len(to_delete) > 0:
        if not bpy.ops.object.delete.poll():
            bpy.ops.object.editmode_toggle()            
        override = bpy.context.copy()
        override["selected_objects"] = to_delete
        with bpy.context.temp_override(**override):
            bpy.ops.object.delete()
        
def copy_and_get_object(obj_name, to_col_name=None):
    """Arg: to_col_name - defines to which collection move object if value is not None"""
    dup = O[obj_name].copy()
    if to_col_name:
        C[to_col_name].objects.link(dup)
    
    return dup

def get_transform_data(obj_name, vertex_group_name):
    transform_data = []
    shnek_obj = D.objects[obj_name]
    place_headers_ix = shnek_obj.vertex_groups[vertex_group_name].index
    for v in shnek_obj.data.vertices:
        for g in v.groups:
            if g.group == place_headers_ix:
                transform_data.append((v.co, v.normal))
    return transform_data

def apply_modifiers(obj, *args, **kwargs):
    # apply wear
    geo_nodes = obj.modifiers.new('wear', 'NODES')
    geo_nodes.node_group = bpy.data.node_groups['Header Wear And Insert']
    geo_nodes['Input_4'] = kwargs['unwear_height']
    geo_nodes['Input_5'] = kwargs['wear_ratio']
    geo_nodes['Input_10'] = kwargs['wear_type']
    geo_nodes['Input_15'] = D.objects['insert_straight top head']
    geo_nodes['Input_16'] = D.objects['insert_straight base']
    geo_nodes['Input_21'] = RGBA_DICT[kwargs['header_color']]
    
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.modifier_apply(modifier=geo_nodes.name, single_user=True)
    
    # add tail
    tail_obj = copy_and_get_object('target tail_raw', 'Headers')
    obj.location[2] += tail_obj.dimensions[2]
    obj.parent = tail_obj
    
    return tail_obj

def set_full_headers(base_object_name, base_vg, col_name, *args, **kwargs):
    shnek_obj = D.objects[base_object_name]            
    transform_data = get_transform_data(base_object_name, base_vg)
    shnek_obj_Y_rot = shnek_obj.rotation_euler[1]

    for pos in transform_data:
        obj = copy_and_get_object(kwargs['header_type'], col_name)
        obj = apply_modifiers(obj, **kwargs)
        to_co = pos[0]
        obj.location = shnek_obj.matrix_world @ to_co
        obj.rotation_mode = "QUATERNION"
        rotation  = Vector((0,0,1)).rotation_difference(pos[1])
        obj.rotation_quaternion = rotation
        
        # CRUTCH
        if round(shnek_obj_Y_rot) == 3:
            obj.rotation_mode = "XYZ"
            obj.rotation_euler[1] -= shnek_obj_Y_rot
        
def render_image(mode, img_ix, save_path=r'C:\Users\user\Documents\Blender\renders'):
    assert mode in ['original', 'mask', 'mask colored'], f'Available modes: {["original", "mask", "mask colored"]}'
    file_format = 'exr' if mode == 'mask colored' else 'png'
    bpy.context.scene.render.filepath = save_path+rf'\{mode}\{img_ix}.{file_format}'
    
    if mode == 'mask colored':
        bpy.context.scene.render.image_settings.file_format = 'OPEN_EXR'
        bpy.context.scene.render.image_settings.color_depth = '32'
        bpy.context.scene.render.image_settings.color_mode = 'BW'
        bpy.context.scene.render.filter_size = 0
        bpy.context.scene.render.dither_intensity = 0
    else:
        bpy.context.scene.render.image_settings.file_format = 'PNG'
        bpy.context.scene.render.image_settings.color_depth = '8'
        bpy.context.scene.render.image_settings.color_mode = 'RGB'
        bpy.context.scene.render.filter_size = 1.5
        bpy.context.scene.render.dither_intensity = 1
    
    engine_dict = {'original': 'CYCLES', 'mask':'BLENDER_EEVEE', 'mask colored':'BLENDER_EEVEE'}
    
    bpy.context.scene.render.engine = engine_dict[mode]
    if engine_dict[mode] == 'CYCLES':
        bpy.context.scene.cycles.device = 'GPU'
    
    bpy.data.scenes["Scene"].node_tree.nodes["isMask"].check = mode != 'original'
    bpy.data.scenes["Scene"].node_tree.nodes["isColorMask"].check = mode == 'mask colored'
    
    start_time = time.time()
    bpy.ops.render.render(write_still = True)
    render_time = time.time() - start_time
    print_console(f'{mode} image render time: {render_time:.2f} secs')


def headers_to_colored_mask():
    for obj in D.collections['Headers'].objects:
        headers_count = len(D.collections['Headers'].objects) // 2
        mod = obj.modifiers.new('set color', 'NODES')
        mod.node_group = bpy.data.node_groups['Set Color by Index']
#        mod['Input_2'] = int(obj.name.split('.')[-1]) - 1
        obj_index = int(obj.name.split('.')[-1]) - 1
        h,s,v = (obj_index / headers_count +0.01, 1, 0.5)
        r,g,b = colorsys.hsv_to_rgb(h, s, v)
        mod['Input_4'] = r # min(r, 0.98) # exclude clear red hue == 0.0
        mod['Input_7'] = g
        mod['Input_8'] = b
            
#            bpy.context.view_layer.objects.active = obj
#            bpy.ops.object.modifier_apply(modifier=mod.name, single_user=True)
        
if __name__ == '__main__':
    configs = pd.read_csv(CONFIGS_FILENAME)
    
#    config = configs.iloc[ix,:]
    
#    shnek_material,\
#    background_type,\

    for config in configs[configs['header_color'] == 'orange'].iloc[315170:315171,:].itertuples():
        delete_all_objects_from_collection('Headers')
        delete_all_objects_from_collection('Tails render')
        ix = config.Index
        # set headers with tails and wear them
        for base_obj_name in ['кривой шнек.001', 'кривой шнек.002', 'прямой шнек']:
            set_full_headers(
                base_obj_name, 
                'Place tails', 
                'Headers',
                header_color=config.header_color,
                wear_type=config.wear_type,
                header_type=config.header_type,
                wear_ratio=config.wear_ratio,
                unwear_height=config.unwear_height
            )
        
        # resetting child-parent relationships
        for obj in D.collections['Headers'].objects:
            if 'header' not in obj.name:
                obj.hide_render = True
                obj.hide_viewport = True
                
                dup = copy_and_get_object(obj.name, 'Tails render')
                dup.hide_render = False
                dup.hide_viewport = False

        for col_name in ['Background']:
            for obj in D.collections[col_name].objects:
                obj.parent = D.objects['WO']
                
        # rotate in given angle
        D.objects['WO'].rotation_euler[1] = config.rotation_angle * pi / 180
        
#        render_image('original', ix)        
        render_image('mask', ix, r'C:\Users\user\Documents\yolo3\data\images')
        headers_to_colored_mask()
        render_image('mask colored', ix, r'C:\Users\user\Documents\yolo3\data\images')


    