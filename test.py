import tensorflow as tf
from V_net import architecture1
import pickle
import os
import argparse
from utils import view_slices_3d


parser = argparse.ArgumentParser(description='Reproducible results')
parser.add_argument('-co', '--component', type=int, default=0, metavar='', help='Field component to plot: 0 - Axial; 1 - Azimuthal; 2 - Radial')
parser.add_argument('-lr', '--init_lr', type=float, default=0.01, metavar='', help='Initial learning rate')
parser.add_argument('-fb', '--filterbase', type=int, default=64, metavar='', help='Number of filter base')
parser.add_argument('-l', '--loss', type=str, default='mse', metavar='', help='Loss function')
parser.add_argument('-sl', '--slice', type=int, default=32, metavar='', help='Sliceeeeee to get the plotting')
parser.add_argument('-prep', '--path_rep', type=str, default="/clusterdata/uqvngu19/scratch/Objective3Datasets/Sample4Rep", metavar='', help='Path to get sample data')
parser.add_argument('-pscaler', '--path_scaler', type=str, default="/clusterdata/uqvngu19/scratch/Objective3Datasets/Combined_train", metavar='', help='Path to get scalers')


args = parser.parse_args()


# -------------------Loading scalers----------------------------------------------
if os.getcwd() != args.path_scaler:
    os.chdir(args.path_scaler)

with open('scaler_axial', 'rb') as scaler_filename:
    scaler_axial = pickle.load(scaler_filename)
with open('scaler_azimuthal', 'rb') as scaler_filename:
    scaler_azimuthal = pickle.load(scaler_filename)
with open('scaler_radial', 'rb') as scaler_filename:
    scaler_radial = pickle.load(scaler_filename)




def main():

    
    optimizer1 = tf.keras.optimizers.Adam(learning_rate=args.init_lr)
    

    model = architecture1(filter_base=args.filterbase)
    model.compile(loss=args.loss, optimizer=optimizer1)

    os.chdir(args.path_rep)

    with open('Geometry4Rep', 'rb') as filename:
        geometry_t = pickle.load(filename)

    with open('Label4Rep', 'rb') as filename:
        label = pickle.load(filename)

    geo_pred = tf.expand_dims(geometry_t, axis=0)
    pred_t = model.predict(geo_pred)
    pred_tt = tf.squeeze(pred_t)


    compo = args.component 

    geometry = geometry_t.numpy()[:, :, :, compo] #sample[index].numpy()[:, :, :, compo]

    axial_l = label.numpy()[:, :, :, compo] #label[index].numpy()[:, :, :, compo]
    # azimuthal_l = label[index].numpy()[:, :, :, 1]
    # radial_l = label[index].numpy()[:, :, :, 2]

    axial_p = pred_tt.numpy()[:, :, :, compo] #pred[index].numpy()[:, :, :, compo]
    # azimuthal_p = pred[index].numpy()[:, :, :, 1]
    # radial_p = pred[index].numpy()[:, :, :, 2]

    t_axial_l = scaler_axial.inverse_transform(axial_l.flatten().reshape((-1, 1)))
    t_axial_p = scaler_axial.inverse_transform(axial_p.flatten().reshape((-1, 1)))

    st_axial_l = t_axial_l.reshape((64, 64, 64))
    st_axial_p = t_axial_p.reshape((64, 64, 64))

    err = st_axial_l - st_axial_p


    view_slices_3d(geometry, st_axial_l, st_axial_p, slice_=args.slice, title='')

if __name__ == '__main__':

    main()

