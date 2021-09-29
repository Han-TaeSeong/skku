import mne



# -----RAW------------- mne.io.RawArray()
# time_as_index()   --> 시간값을 Index 값으로  ex)38초 -> 96250
#
# ch_names()      --> 채널 이름보기
# drop_channels([list])     -->  채널없애기
# pick_channels([list])     -->  채널선택하기
# crop(tmax,tmin)         --> 출력시간 정하기
#
# rename_channels(mapping) # channel 이름 변경
# set_montage('standard_1020') # 10-20 system에 적용
#
# to_data_frame()         --> pandas
# get_data            --> numpy
# np.save(file=,arr=)         -->save
#
# set -->
#
# plot(n_channels=10,scalings=100)   --> n_channels: 한 번에 볼 채널 수
# plot_psd(fmax=30)       --> fmax:최대 30Hz까지 나타냄 //Power Spectral Density






### channel maping ###
mapping = {
    '0': 'Fz', '1': 'FC3', '2': 'FC1', '3': 'FCz', '4': 'FC2',
    '5': 'FC4', '6': 'C5', '7': 'C3', '8': 'C1', '9': 'Cz',
    '10': 'C2', '11': 'C4', '12': 'C6', '13': 'CP3',
    '14': 'CP1', '15': 'CPz', '16': 'CP2', '17': 'CP4', '18': 'P1',
    '19': 'Pz', '20': 'P2', '21': 'POz', '22': 'Fp1', '23': 'Fpz', '24': 'Fp2'
} # channel 이름 설정
b.rename_channels(mapping) # channel 이름 변경
b.set_montage('standard_1020') # 10-20 system에 적용



#### ICA(independent component analysis) ###
# detection blinks and generating corresponding events
eog_evoked = create_eog_epochs(b, ch_name='Fp1,Fp2,Fpz').average()
eog_evoked.apply_baseline(baseline=(None, -0.1))
eog_evoked.plot_joint()
plt.show()

# # low frequency drift(slow drift) high-pass filter를 통해 제거
raw = b.copy()
# raw.load_data().filter(l_freq=1., h_freq=None) # slow drift 제거 (1Hz 성분 제거, high-pass filter)

# Run ICA
ica = ICA(n_components=15, max_iter='auto', random_state=97) # component num 결정
ica.fit(raw) # Run the ICA decomposition on raw data.

# ICA components에 대한 data를 plot한다.
b.load_data()
ica.plot_sources(b, show_scrollbars=False)

# plot ICA components
ica.plot_components()

# EOG와 비슷한 components 확인
ica.plot_overlay(b, exclude=[1], picks='eeg')

# EOG와 비슷한 components 분석
ica.plot_properties(b, picks=[1])
plt.show()



### components에서 EOG 비교 (matching) ###
ica.exclude = []
# find which ICs match the EOG pattern
eog_indices, eog_scores = ica.find_bads_eog(b, ch_name='Fp1,Fpz,Fp2')
ica.exclude = eog_indices

# barplot of ICA component "EOG match" scores
ica.plot_scores(eog_scores)

# plot diagnostics
ica.plot_properties(b, picks=eog_indices)

# plot ICs applied to raw data, with EOG matches highlighted
ica.plot_sources(b, show_scrollbars=False)

# plot ICs applied to the averaged EOG epochs, with EOG matches highlighted
ica.plot_sources(eog_evoked)
plt.show()



### ICA 결과 확인 ###
ica.exclude = [] # ICA component 제거
ica.exclude = eog_indices # ICA EOG component로 설정
mne.preprocessing.ICA.save # ICA 저장
print(ica)
ica.apply(b) # ICA 적용
b.plot(n_channels=25, scalings=100) # EOG artifact 제거 확인
plt.show()