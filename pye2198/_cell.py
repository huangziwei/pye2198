import os
import pickle
import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd

__all__ = ['Cell']

class Cell:
    
    def __init__(self, cell_id, file_directory='../data/'):
        
        self.filename=cell_id

        cell_data = get_cell_data(cell_id, file_directory)

        self.voxelsize = np.array([92., 66., 66.]) / 1000

        self.index = cell_data['index'] # index of each node
        self.edges = cell_data['edges'] # Connectivity of skeleton nodes 
        self.coord = cell_data['coord'] * self.voxelsize # Coordinates of skeleton nodes (N x 3)
        self.radii = cell_data['radii']
        self.coord_types = cell_data['coord_types']
        self.point_cloud = cell_data['point_cloud'] * self.voxelsize

        G = nx.DiGraph()
        G.add_nodes_from(self.index)
        G.add_edges_from(self.edges)
    
        self.branch_coord = [i for i in G.nodes if G.degree(i) == 3]
        self.terminal_coord = [i for i in G.nodes if G.degree(i) == 1]

        self.G = G
        
    def get_paths(self):
        
        self.df_paths = _get_paths(self.G, self.terminal_coord, self.branch_coord, self.coord, self.radii, self.coord_types)        
        
    def plot_morph(self, zoom=False, highlight=[]):

        df_paths = self.df_paths
        
        fig = plt.figure(figsize=(10,10))
        gs = fig.add_gridspec(3, 3)

        ax0 = fig.add_subplot(gs[:2, :2])
        ax1 = fig.add_subplot(gs[:2, 2])
        ax2 = fig.add_subplot(gs[2, :2])

        for row in df_paths.iterrows():

            idx = row[0]
            path = row[1]['path']

            if idx in highlight:

                ax0.plot(path[:, 0], path[:, 1], color='C1')
                ax1.plot(path[:, 2], path[:, 1], color='C1')
                ax2.plot(path[:, 0], path[:, 2], color='C1')

            else:

                ax0.plot(path[:, 0], path[:, 1], color='grey')
                ax1.plot(path[:, 2], path[:, 1], color='grey')
                ax2.plot(path[:, 0], path[:, 2], color='grey')

        ax0.scatter(self.coord[0][0], self.coord[0][1], s=200, color='C0')
        ax1.scatter(self.coord[0][2], self.coord[0][1], s=200, color='C0')
        ax2.scatter(self.coord[0][0], self.coord[0][2], s=200, color='C0')

        if zoom:

            ax0.set_xlim(-15*2, 15*2)
            ax0.set_ylim(-15*2, 15*2)

            ax1.set_xlim(5*2, -5*2)
            ax1.set_ylim(-15*2, 15*2)

            ax2.set_xlim(-15*2, 15*2)
            ax2.set_ylim(-5*2, 5*2)

        else:

            ax0.set_xlim(-150, 150)
            ax0.set_ylim(-150, 150)

            ax1.set_xlim(50, -50)
            ax1.set_ylim(-150, 150)

            ax2.set_xlim(-150, 150)
            ax2.set_ylim(-50, 50)

        for ax in [ax0, ax1, ax2]:
            ax.set_xticks([])
            ax.set_yticks([])
            
    def find_paths_connect_to_soma(self):

        df = self.df_paths
        soma = self.coord[0]

        candidates = []
        print('Potential paths connected to soma:\n')
        for counter, scale_factor in enumerate([1, 1.5, 2, 2.5, 3]):

            a = df['path'].apply(lambda x: distance_between_two_points(x[0], soma)) # one end
            b = df['path'].apply(lambda x: distance_between_two_points(x[-1], soma)) # the other end

            shortest_dist = np.minimum(a.min(),b.min())
            indxa = df[a < shortest_dist * scale_factor].index.values
            indxb = df[b < shortest_dist * scale_factor].index.values

            candidates.append(np.unique(np.hstack([indxa, indxb])))

            print(f'  [{counter}] #{len(candidates[-1])}: {candidates[-1]}')
        self.candidates = candidates
        
    def add_soma_to_path(self, path_ids):

        df = self.df_paths
        soma = self.coord[0] 

        df['connect_to_at'] = ''
        df['connect_to_at'] = df['connect_to_at'].apply(np.array)

        distance_matrix = np.vstack(df['path'].loc[path_ids].apply(
            lambda x: [
                       distance_between_two_points(x[0], soma), 
                       distance_between_two_points(x[-1], soma)
                      ]
            ).values)

        reverse_or_not = -(distance_matrix[:, 0] > distance_matrix[:, 1]).astype(int)

        path_ids_to_reverse = path_ids[reverse_or_not==-1]

        for path_id_to_reverse in path_ids_to_reverse:
            df.at[path_id_to_reverse, 'path'] = df.at[path_id_to_reverse, 'path'][::-1]
            df.at[path_id_to_reverse, 'nodes'] = df.at[path_id_to_reverse, 'nodes'][::-1]
            df.at[path_id_to_reverse, 'nodes_downsampled'] = df.at[path_id_to_reverse, 'nodes_downsampled'][::-1]
            
        for path_id in path_ids:

            df.at[path_id, 'path'] = np.vstack([soma, df.at[path_id, 'path']])
            df.at[path_id, 'nodes'] = np.hstack([0, df.at[path_id, 'nodes']])
            df.at[path_id, 'nodes_downsampled'] = np.hstack([0, df.at[path_id, 'nodes_downsampled']])
            df.at[path_id, 'connect_to'] = -1
            df.at[path_id, 'connect_to_at'] = soma

        self.df_paths = df
        
        
    def connect_all_paths(self, path_ids):

        df = self.df_paths

        df_connected = df.loc[path_ids]
        df_to_connect = df.loc[~df.index.isin(path_ids)]

        paths_checked = set()
        paths_to_check = set(df_connected.index.values.tolist())

        while len(paths_to_check) > 0:

            for current_path_id in paths_to_check:

                # get the tail of the current path
                tail = df_connected.loc[current_path_id].nodes[-1] 

                # check if there are paths that connect to the tail directly
                path_ids_to_connect = df_to_connect[df_to_connect['nodes'].apply(lambda x: x[0] == tail)].index.values

                # a path with branchpoint should be connected by two paths
                # if paths_ids_to_connect is less than two, try reverse all those paths
                if len(path_ids_to_connect) < 2:

                    more_paths_id_to_connect = df_to_connect[df_to_connect['nodes'].apply(lambda x: x[-1] == tail)].index.values
                    path_ids_to_connect = np.hstack([path_ids_to_connect, more_paths_id_to_connect])

                    for path_id_to_reverse in more_paths_id_to_connect:
                        df_to_connect.at[path_id_to_reverse, 'path'] = df_to_connect.at[path_id_to_reverse, 'path'][::-1]
                        df_to_connect.at[path_id_to_reverse, 'nodes'] = df_to_connect.at[path_id_to_reverse, 'nodes'][::-1]

                # what if there are still less than two?
                # need something more here

                df_to_connect.is_copy = False
                df_to_connect.at[path_ids_to_connect, 'connect_to'] = current_path_id
                df_to_connect.at[path_ids_to_connect, 'connect_to_at'] = df_to_connect.loc[path_ids_to_connect, 'path'].apply(lambda x: x[0]).values
                # add those paths to df_connected
                df_connected = df_connected.append(df_to_connect.loc[path_ids_to_connect])

                # remove those paths from df_to_connect
                df_to_connect = df_to_connect.drop(path_ids_to_connect)

                paths_checked.add(current_path_id)

            paths_to_check = set(df_connected.index.values.tolist())
            paths_to_check = paths_to_check - paths_checked

        # final check to connect all the disjoint paths
        for row in df_to_connect.iterrows():

            path_id = row[0]
            path = row[1]['path']

            a = df_connected.path.apply(lambda x: distance_between_two_points(x[-1], path[0])) # one end
            b = df_connected.path.apply(lambda x: distance_between_two_points(x[-1], path[-1])) # the other end

            if np.argmin([a.min(), b.min()]) == 0:

                path_id_connected_to = a.idxmin()

            else:

                path_id_connected_to = b.idxmin()
                path = path[::-1]

            df_to_connect.at[path_id, 'path'] = path
            df_to_connect.at[path_id, 'connect_to'] = path_id_connected_to
            df_to_connect.at[path_id, 'connect_to_at'] = df_connected.loc[path_id_connected_to].path[-1]
            df_connected = df_connected.append(df_to_connect.loc[path_id])
            df_to_connect = df_to_connect.drop(path_id)        

        self.df_connected = df_connected
        self.df_to_connect = df_to_connect
        
        self.df_paths = self.df_connected.sort_index()
        
    def get_radius(self):

        point_cloud = self.point_cloud
        all_coord = self.coord
        all_nodes = np.unique(np.hstack(self.df_paths['nodes_downsampled']))
        
        vol_voxel = np.prod(self.voxelsize)

        radii = np.zeros(len(self.coord))
        
        print('\tRadius\tRatio')
        for counter, idx in enumerate(all_nodes):
            
            ratio0 = []
            
            print(f"{idx}:{counter}/{len(all_nodes)}")   
            for r in np.arange(0.1, 5, 0.033):
                
                vol_sphere = 4 * np.pi * r **3 / 3

                points = point_cloud[np.sqrt(np.sum((point_cloud - all_coord[idx]) ** 2, 1)) <= r]
                vol_points = len(points) * vol_voxel

                ratio0.append(vol_points / vol_sphere)

                print(f'\t{r}\t {vol_points:.03f}/{vol_sphere:.03f}={ratio0[-1]:.03f}')
       
                if r == 0.1 and ratio0[-1] < 0.95:
                    radii[idx] = r
                    print(f'\n\tFinal: radius={radii[idx]} | ratio={ratio0[-1]:.03f}\n')
                    break
                elif ratio0[-1] <=0.95:
                    radii[idx] = r-0.033
                    print(f'\n\tFinal: radius={radii[idx]} | ratio={ratio0[-2]:.03f}\n')
                    break
            
        radii[0] = self.radii[0]
        self.radii = radii    
        
    def finalize(self):

        df_paths = self.df_paths
        # find all paths connect to current path.
        connected_by_dict = {}
        connected_by_at_dict = {}
        for path_id in df_paths.index:
            connected_by_dict[path_id]    = df_paths[df_paths.connect_to == path_id].index.tolist()
            connected_by_at_dict[path_id] = df_paths[df_paths.connect_to == path_id].connect_to_at.tolist()
        df_paths['connected_by'] = pd.Series(connected_by_dict)
        df_paths['connected_by_at'] = pd.Series(connected_by_at_dict)
        
        back_to_soma_dict = {}
        for path_id in df_paths.index:
            list_to_soma = [path_id]
            next_path_id = df_paths.loc[path_id].connect_to
            while next_path_id != -1:
                list_to_soma.append(next_path_id)
                next_path_id = df_paths.loc[next_path_id].connect_to
            back_to_soma_dict[path_id] = list_to_soma
        df_paths['back_to_soma'] = pd.Series(back_to_soma_dict)
        
        self.df_paths = df_paths

        self.df_paths['radius'] = self.df_paths['nodes_downsampled'].apply(lambda x: self.radii[x])
        self.df_paths['types'] = self.df_paths['nodes_downsampled'].apply(lambda x: self.coord_types[x])

    def export_swc(self, save_to='../output/'):

        df_paths = self.df_paths
        
        path_checked = []

        soma_coord = df_paths.loc[df_paths['connect_to'] == -1].path.iloc[0][0]
        soma_radius = df_paths.loc[df_paths['connect_to'] == -1].radius.iloc[0][0]
        swc_arr = np.array([[1, 1, soma_coord[0], soma_coord[1], soma_coord[2], soma_radius, -1]])
        # ['n', 'type', 'x', 'y', 'z', 'radius', 'parent']    

        list_back_to_soma = (df_paths.sort_values(['connect_to']).back_to_soma).tolist()
            
        for i, back_to_soma in enumerate(list_back_to_soma):

            for path_id in back_to_soma[::-1]:

                if path_id in path_checked: 
                    continue
                
                path_data = df_paths.loc[path_id]
                path = path_data['path'][1:]
                path_radius = path_data['radius'][1:]
                path_type = path_data['types'][-1]  

                connect_to = path_data['connect_to']
                connect_to_at = path_data['connect_to_at']

                swc_path = np.column_stack([np.ones(len(path)) * path_type, path]) # type
                swc_path = np.column_stack([np.arange(len(swc_arr)+1, len(path)+len(swc_arr)+1), swc_path]) #ID
                swc_path = np.column_stack([swc_path, path_radius * np.ones(len(path))]) # radius
                swc_path = np.column_stack([swc_path, swc_path[:, 0]-1]) # placeholder for PID
    #             print(i, path_id)
    #             return swc_path
                pid = np.where((swc_arr[:, 2:5] == connect_to_at).all(1))[0] + 1
                if len(pid) > 1:
                    swc_path[0][-1] = pid[0]
                else:
                    swc_path[0][-1] = pid
                
                swc_arr = np.vstack([swc_arr, swc_path])
                path_checked.append(path_id)
                
        df_swc = pd.DataFrame(swc_arr)
        df_swc.index = np.arange(1, len(df_swc)+1)
        df_swc.columns = ['n', 'type', 'x', 'y', 'z', 'radius', 'parent']
        df_swc[['n', 'type', 'parent']] = df_swc[['n', 'type', 'parent']].astype(int)
     
        self.df_swc = df_swc
        self.df_swc.to_csv(save_to + 'Cell_{}.swc'.format(self.filename), sep=' ', index=None, header=None)        
        
def get_cell_data(cell_id, file_directory):

    cell_info = scipy.io.loadmat(file_directory + '/data/cell_info.mat')['cell_info']
    
    mat = scipy.io.loadmat(file_directory + '/skeletons_GC/skel_{}'.format(cell_id))
    
    edges = mat['e'].astype(int)
    radii = mat['rad'].astype(float) # radius
    coord = mat['n'].astype(float) # node
    point_cloud = mat['p'].astype(float) # point cloud

    soma_pos, soma_size = get_soma_coord(cell_id, cell_info)
    coord = np.vstack([soma_pos, coord])
    coord -= soma_pos
    point_cloud -= soma_pos
    
    index = np.arange(len(coord))
    radii = np.vstack([np.sqrt(soma_size / np.pi), radii])
    coord_types = np.ones(len(coord)) * 3
    coord_types[0] = 1
    
    return {
            'index': index,
            'coord': coord,
            'edges': edges,
            'radii': np.ravel(radii),
            'coord_types': np.ravel(coord_types),  
            'point_cloud': point_cloud, # point cloud 
    }

def get_soma_coord(cell_id, cell_info):
    for i in range(1789):
        if cell_info[i][0][0] == cell_id:
            soma = cell_info[i][0][9][0][::-1]
            somasize = cell_info[i][0][12][0][0]
            break
    return soma, somasize

def check_head(G, head):
    
    """
    Check if the head node is a good starting point
    for get_segment()
    """
    
    if G.degree(head) == 3: 
        # if starting node is a branch point, 
        # should move it to the next point
        # first check if the predecessors can be the starting node
        node = list(G.predecessors(head))
        if len(node) == 2:
            # if there are two predecessors, then use successors
            head = G.successors(node)
            mode = 1
            return head, mode
        else:
            # branch points are always has predecessors. 
            head = node[0]
            mode = 0 
            return head, mode
    else:
        # if a head is not branch point, then it's a terminal point
        # check if there is a predecessor
        node = list(G.predecessors(head))
        if len(node) == 0:
            # if there's no predecssor, use successors
            mode = 1
            return head, mode
        else:
            mode = 0
            return head, mode
            
    
def get_segment_nodes(G, head, mode=0, count=0):
    
    """
    Get head and tail node index of one segment.
    
    A segment is defined as a neurite path between one terminal node to
    a branch node or between two branch nodes.
    
    Parameters
    ----------
    G : networkx Graph object
    
    head : int 
        starting node idx
    
    mode: int
        if mode==0, use G.predecessors 
        if mode==1, use G.successors
    
    Returns
    -------
    tail : int
        ending node idx
    """
    
    if count == 0:
        node, mode = check_head(G, head)
        count+=1
    else:
        node = head
        count+=1
    
    if mode == 0: # use predecessors
        get_next_node = G.predecessors
    elif mode == 1: # use successors 
        get_next_node = G.successors
    
    if G.degree(node) != 3:
        
        last_node = node
        node = list(get_next_node(node))
        if len(node) > 0:
            # successfully reach to a next node or a branch point
            [node, tail], mode = get_segment_nodes(G, node[0], mode, count)
        else:
            # next node is empty means that it's a losse end path
            tail = last_node
    else:
        tail = head
        
    node = head
    return [head, tail], mode 

def get_res(G, tn, bn, p, r, t):
    
    res = []
    for head in tn + bn:
        [hd, tl], m = get_segment_nodes(G, head)

        if m == 0:
            res.append([hd,tl])
        else:
            res.append([tl,hd])

    res = np.array(res)
    res = np.unique(res, axis=0)

    return res

def _get_paths(G, tn, bn, p, r, t):

    """
    return a pandas DataFrame with two columns
    nodes: index of all nodes of the segment
    path: voxel coordinates of all nodes of the segment

    """
    
    res = []
    for head in tn + bn:
        [hd, tl], m = get_segment_nodes(G, head)

        if m == 0:
            res.append([hd,tl])
        else:
            res.append([tl,hd])

    res = np.array(res)
    res = np.unique(res, axis=0)
    res = res[~(abs(res[:,0] - res[:,1]) < 2)] # not sure
    # res = np.sort(res, 0)
    
    all_paths_idx = {}
    for idx, rr in enumerate(res):
        paths_between = nx.all_simple_paths(G,source=rr[1],target=rr[0])
        nodes_between = [node for path in paths_between for node in path]
        all_paths_idx[idx] = nodes_between
        

    df = pd.DataFrame()
    df['nodes'] = pd.Series(all_paths_idx)
    df['nodes_downsampled'] = df.nodes.apply(lambda x: downsample(x, 10))
    df['path'] = df['nodes_downsampled'].apply(lambda x: p[x])
    
    return df

def distance_between_two_points(a, b):
    return np.sqrt(np.sum((a-b)**2))

def downsample(arr, scale):
    
    head = arr[0]
    tail = arr[-1]
    mid = arr[1:-1][::scale]
    
    return np.hstack([head, mid, tail])