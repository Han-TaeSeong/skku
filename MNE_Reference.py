import mne



-----RAW------------- mne.io.RawArray()
time_as_index()   --> 시간값을 Index 값으로  ex)38초 -> 96250

ch_names()      --> 채널 이름보기
drop_channels([list])     -->  채널없애기
pick_channels([list])     -->  채널선택하기
crop(tmax,tmin)         --> 출력시간 정하기

to_data_frame()         --> pandas
get_data            --> numpy
np.save(file=,arr=)         -->save

set -->

plot(n_channels=10,scalings=100)   --> n_channels: 한 번에 볼 채널 수
plot_psd(fmax=30)       --> fmax:최대 30Hz까지 나타냄 //Power Spectral Density