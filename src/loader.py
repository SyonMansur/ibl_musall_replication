import numpy as np
from scipy.interpolate import interp1d
from brainbox.io.one import SpikeSortingLoader
from one.api import ONE

class MusallDataLoader:
    def __init__(self):
            # standard initialization       
            self.one = ONE(
                base_url='https://openalyx.internationalbrainlab.org', 
                password='international', 
                silent=True # silent so the terminal doesn't get spammed
            )
        
    def process_sessions(self, eid, bin_size = 0.05):
        """
        load wheel and neural data, align them
        take in experiment uuid and bin width (seconds)

        returns:
            wheel_vel and spike_counts
        """

        try:
            # load the wheel data using a single recording session (eid)
            wheel = self.one.load_object(eid, 'wheel', collection = 'alf')
        except Exception as e:
            print(f"error loading wheel: {e}")
            return None, None
        
        # get the pid associated with the eid (individual probe)
        pids, _ = self.one.eid2pid(eid)

        if not pids:
            print("no probes found")
            return None, None
        
        pid = pids[0] # for now just first probe
        print(f"using probe {pid}")

        sl = SpikeSortingLoader(pid=pid, one=self.one)
        # download the spike timings and clusters (which neuron fired)
        spikes, clusters, channels = sl.load_spike_sorting()
        # merge the metrics with the channel locations
        clusters = sl.merge_clusters(spikes, clusters, channels)

        # now we need to filter the neurons
        good_cluster_mask = clusters['label'] == 1 # should be a good neuron
        good_ids = clusters['cluster_id'][good_cluster_mask] # mask

        if len(good_ids) == 0:
            print("no good ones :(")
            return None, None
        
        # now let's algin everything to the same times
        t_start = wheel.timestamps[0] # first timestamp
        t_end = wheel.timestamps[-1] # last timestamp
        t_bins = np.arange(t_start, t_end, bin_size)

        # so that the irregular wheel data has continuous values; interpolation
        f_wheel = interp1d(wheel.timestamps, wheel.position, kind = 'linear', fill_value = 'extrapolate') # gotta guess sometimes
        wheel_pos_binned = f_wheel(t_bins)

        # get velocity by getting differences in position
        wheel_vel = np.diff(wheel_pos_binned, prepend=wheel_pos_binned[0]) # prepend to keep shape matching tbins
        # (shrinks array by 1)


        # bin
        spike_mask = np.isin(spikes['clusters'], good_ids)
        filtered_times = spikes['times'][spike_mask]
        filtered_clusters = spikes['clusters'][spike_mask]

        # vectorization; get a matrix 2x2 of time, neurons; using histogram
        spike_counts, _, _ = np.histogram2d(
            filtered_times, 
            filtered_clusters, 
            bins=[t_bins, np.r_[good_ids, good_ids[-1]+1]] # bin edges trick
        )
        
        # trim just in case histogram2d adds an extra edge
        if spike_counts.shape[0] > len(wheel_vel):
            spike_counts = spike_counts[:len(wheel_vel)]

        print(f"done. aligned {len(wheel_vel)} bins.")
        return wheel_vel, spike_counts
    
if __name__ == "__main__":
    loader = MusallDataLoader()
    
    # mouse DY_018 (should work)
    eid = '4b00df29-3769-43be-bb40-128b1cba6d35'
    
    print(f"loading session: {eid}")
    
    vel, neurons = loader.process_sessions(eid)

    if vel is not None:
        print(f"success.")
        print(f"  vel shape: {vel.shape}")
        print(f"  neur shape:  {neurons.shape}")
    else:
        print("broken")