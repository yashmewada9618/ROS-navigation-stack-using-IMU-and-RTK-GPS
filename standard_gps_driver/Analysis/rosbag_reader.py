from bagpy import bagreader
import pandas as pd
import seaborn as sea
import matplotlib.pyplot as plt
import numpy as np

class bag_file_reader:
    def __init__(self,bag_file):
        my_bag = bagreader(bag_file)
        self.my_topic = my_bag.message_by_topic('/gps')
    
    def scatter_plot(self,x,y,dataset,xlabel,ylabel,label,title):
        sea.scatterplot(y=y, x=x, data=dataset,label = label,sizes=[100],marker='x',legend=False)
        plt.xlabel(xlabel,fontsize=20)
        plt.ylabel(ylabel,fontsize=20)
        plt.title(title,fontsize=18)
        plt.legend(labels=['stat open','stat occ','mov occ','mov open'])
        filename = 'plot-' + str(title) +'.png'
        plt.savefig(filename)

    def read_plot_utm(self,i):
        data = pd.read_csv(self.my_topic,usecols = ['UTM_easting','UTM_northing'])
        first_value_e = data['UTM_easting'][0]
        data['Scaled_Easting'] = data['UTM_easting'] - first_value_e
        first_value = data['UTM_northing'][0]
        data['Scaled_Northing'] = data['UTM_northing'] - first_value
        x,y,dataset,xlabel,ylabel,label,title= data['Scaled_Easting'],data['Scaled_Northing'],data,'Easting (M)','Northing (M)','UTM_East/North','Scatterplot of UTM Easting vs Northing'
        self.scatter_plot(x,y,dataset,xlabel,ylabel,label,title + str(i))
        return data
    
    def utm_line_fit(self,i):
        data = pd.read_csv(self.my_topic,usecols = ['UTM_easting','UTM_northing'])
        first_value_e = data['UTM_easting'][0]
        data['Scaled_Easting'] = data['UTM_easting'] - first_value_e
        first_value = data['UTM_northing'][0]
        data['Scaled_Northing'] = data['UTM_northing'] - first_value
        x,y,dataset,xlabel,ylabel,label,title= data['Scaled_Easting'],data['Scaled_Northing'],data,'Easting (M)','Northing (M)','UTM_East/North','Line fitting over UTM data'
        self.scatter_plot(x,y,dataset,xlabel,ylabel,label,title + str(i))
        sea.regplot(x = data['Scaled_Easting'],y = data['Scaled_Northing'])
        plt.xlabel(xlabel,fontsize=20)
        plt.ylabel(ylabel,fontsize=20)
        filename = 'plot-' + str(title) + str(i) +'.png'
        plt.savefig(filename)
        a,b = np.polyfit(data['Scaled_Easting'],data['Scaled_Northing'],1)
        predicted_northing = a*data['Scaled_Easting'] + b
        residual_error = data['Scaled_Northing'] - predicted_northing
        rmse_line_fit = np.sqrt(np.mean(residual_error**2))
        return data,rmse_line_fit
    
    def lat_long_line_fit(self):
        data = pd.read_csv(self.my_topic,usecols = ['Latitude','Longitude'])
        first_value_e = data['Latitude'][0]
        data['Scaled_Easting'] = data['Latitude'] - first_value_e
        first_value = data['Longitude'][0]
        data['Scaled_Northing'] = data['Longitude'] - first_value
        x,y,dataset,xlabel,ylabel,label,title= data['Scaled_Easting'],data['Scaled_Northing'],data,'Easting (M)','Northing (M)','UTM_East/North','Scatterplot of UTM Easting vs Northing'
        self.scatter_plot(x,y,dataset,xlabel,ylabel,label,title)
        sea.regplot(x = data['Scaled_Easting'],y = data['Scaled_Northing'])
        a,b = np.polyfit(data['Scaled_Easting'],data['Scaled_Northing'],1)
        predicted_northing = a*data['Scaled_Easting'] + b
        residual_error = data['Scaled_Northing'] - predicted_northing
        rmse_line_fit = round(np.sqrt(np.mean(residual_error**2)),5)
        return data,rmse_line_fit
    
    def read_ltlg(self,known_pos):
        data = pd.read_csv(self.my_topic,usecols = ['Latitude','Longitude'])
        # known_utm = [327827.25,4689374.84]
        data['lat_error'] = data['Latitude'] - known_pos[0]
        data['long_error'] = data['Longitude'] - known_pos[1]
        sea.histplot(data['lat_error'],bins=50,color='orange')
        # filename = 'plot-' + str('lat_error') +'.png'
        # plt.savefig(filename)
        # plt.show()
        sea.histplot(data['long_error'],bins=50)
        plt.xlabel('Error',fontsize=20)
        plt.ylabel('Count',fontsize=20)
        plt.title('Histogram for stat occluded',fontsize=18)
        plt.legend(labels=['lat_error','long_error'])
        filename = 'plot-' + str('Histogram for stat occluded') +'.png'
        plt.savefig(filename)
        # plt.savefig(filename)
        plt.show()
        return data
    
    def cal_RMSE(self,pos):
        lt_lg = pd.read_csv(self.my_topic,usecols = ['Latitude','Longitude'])
        print(lt_lg)
        lat_rmse = 0
        long_rmse = 0
        for (col_name,col_value) in lt_lg.iteritems():
            if col_name == 'Latitude':
                lat_rmse += (col_value.values - pos[0])**2
            elif col_name == 'Longitude':
                long_rmse += (col_value.values - pos[1])**2
        lat_rmse = lat_rmse.mean()**0.5
        long_rmse = long_rmse.mean()**0.5
        singular_rmse = (lat_rmse**2 + long_rmse**2)**0.5
        return lat_rmse,long_rmse,singular_rmse
    
    def cal_utm_rmse(self,pos):
        utm_dat = self.read_plot_utm(1)
        utm_e = utm_dat['UTM_easting'].std()
        utm_n = utm_dat['UTM_northing'].std()
        CEP = 0.59*(utm_e + utm_n)
        RMS = (utm_e**2 + utm_n**2)**0.5
        # utm_n = utm_dat['UTM_northing'].mean()
        utm_e_sigma = 0.0
        utm_n_sigma = 0.0
        for (col_name,col_value) in utm_dat.iteritems():
            if col_name == 'UTM_easting':
                utm_e_sigma += (col_value.values - pos[0])**2
            elif col_name == 'UTM_northing':
                utm_n_sigma += (col_value.values - pos[1])**2
        utm_e_sigma = utm_e_sigma.mean()**0.5
        utm_n_sigma = utm_n_sigma.mean()**0.5
        singular_utm_rmse = (utm_e_sigma**2 + utm_n_sigma**2)**0.5
        return utm_e_sigma,utm_n_sigma,singular_utm_rmse
    
    def read_alt(self,i):
        data = pd.read_csv(self.my_topic,usecols = ['Altitude','Header.stamp.secs'])
        first_data = data['Header.stamp.secs'][0]
        data['scaled_time'] = data['Header.stamp.secs'] - first_data
        x,y,dataset,xlabel,ylabel,label,title = data['scaled_time'],data['Altitude'],data,'Time (Sec)','Altitude (M)','Altitude','Scatterplot of Altitude vs Time'
        self.scatter_plot(x,y,dataset,xlabel,ylabel,label,title + str(i))
        return data

    
if __name__ == '__main__':
    bag_file_path1 = '/home/yash/Documents/LAB1/src/gps_driver/Data/stationary.bag'
    bag_file_path2 = '/home/yash/Documents/LAB1/src/gps_driver/Data/Stationary_open.bag'
    bag_file_path3 = '/home/yash/Documents/LAB1/src/gps_driver/Data/moving_straight.bag'
    bag_file_path4 = '/home/yash/Documents/LAB1/src/gps_driver/Data/moving_back.bag'
    
    stat = bag_file_reader(bag_file_path1)
    stat_open = bag_file_reader(bag_file_path2)
    mov_occ = bag_file_reader(bag_file_path3)
    mov_open = bag_file_reader(bag_file_path4)

    stat_pos = [42.3374628,-71.0900613] #for stationary occluded
    stat_pos_utm = [327820.07,4689361.02]
    # stat_open_pos = [42.337066, -71.090451]
    # mov_occ_pos = [42.337753, -71.089887] # for walking ahead
    # known_utm = [327835.20, 4689392.87]
    # known_utm1 = [327827.25,4689374.84]
    stat.read_ltlg(stat_pos)
    # stat_open.read_plot_utm(" stat open")
    # mov_occ.read_plot_utm(" mov occ")
    # mov_open.read_plot_utm(" all data")
    # mov_open.utm_line_fit(" mov open")
    # mov_occ.utm_line_fit(" compare")
    # print(stat.cal_utm_rmse(stat_pos_utm))
    # print(stat.cal_RMSE(stat_pos))
    # stat.read_alt(" stat occ")
    # stat_open.read_alt(" stat compare")
    
        # plt.legend(labels=['move occ','move open'])
    # mov_occ.read_alt(" mov occ")
    # mov_open.read_alt(" mov compare")
    plt.show()