from shapely.geometry import MultiPolygon
from shapely.geometry import LineString
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import matplotlib.animation as animation
def render_relative_map(map_dict):
    """_summary_

    Args:
        map_dict (_type_): label : [data0,data1,data2,...]
    """
    cnt = 0
    fig,ax = plt.subplots()
    for item in map_dict:
        for i,shapely_obj in enumerate(map_dict[item]):
            if isinstance(shapely_obj,MultiPolygon):
                for j, polygon in  enumerate(shapely_obj):
                    xy= np.array(polygon.exterior.xy).T
                    pg = patches.Polygon(xy,closed=True,alpha=0.5,fill=None,edgecolor=f'C{cnt}')
                    if i+j == 0:
                        pg.set_label(item)
                    ax.add_patch(pg)
            elif isinstance(shapely_obj,LineString):
                x,y= shapely_obj.xy
                ax.plot(x,y,alpha=0.5,color=f'C{cnt}',label=item if i == 0 else "")
            else:
                print("?")
        cnt = cnt+1
    ax.autoscale_view()
    ax.legend()
    plt.show()
            
class RenderDataset:
    def __init__(self,key:str = ''):
        self.fig, self.ax = plt.subplots()
        self.key = key
        if key == '':
            self.key='geoms_dict'

    def update(self,frame):
        
        cnt = 0
        artist_obj_list= []
        geoms_dict = frame[self.key]
        for item in geoms_dict:

            for i,shapely_obj in enumerate(geoms_dict[item]):
                if isinstance(shapely_obj,MultiPolygon):
                    for j, polygon in  enumerate(shapely_obj):
                        xy= np.array(polygon.exterior.xy).T
                        pg = patches.Polygon(xy,closed=True,alpha=0.5,fill=None,edgecolor=f'C{cnt}')
                        artist_obj_list.append( self.ax.add_patch(pg))
                elif isinstance(shapely_obj,LineString):
                    x,y= shapely_obj.xy
                    line, = self.ax.plot(x,y,alpha=0.5,color=f'C{cnt}')
                    artist_obj_list.append(line)
                else:
                    print("unknown: "+item)
                    # print(geoms_dict[item])
            cnt = cnt+1
        self.ax.autoscale_view()
        return artist_obj_list

    def render(self,dataset):
        self.ani = animation.FuncAnimation(self.fig, self.update, frames=dataset,blit=True, interval=500, repeat=True)
        plt.show()
