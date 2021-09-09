# skku


BCIMI(Brain-Computer-Interface-Motor-Imagery) Dataset 사용

Dataset 출처
http://www.bbci.de/competition/iv/#dataset1
Data sets 2a: ‹4-class motor imagery›



Dataset의 구성
4가지의 동작을 상상했을 때 뇌파를 측정.
class1 = 왼손
class2 = 오른손
class3 = 양발
class4 = 혀

총 6개의 session
각각의 session 안에서 48trials 측정(각 class 마다 12trials)
총 6X48 = 288 trails

Session의 구성
5분간 EOG 측정
(2m: Eyes open // 1m: Eyes closed //2: Movement)
(A04T only have Movement //technical issue)

At the beginning of a trial
t=0 고정된 십자가 모니터에 나온다 and 삐 소리(cue sign)
2<t<3.25 [상(혀), 하(양발), 좌(왼손), 우(오른손)]를 가르키는 화살표 등장
3~t<6 움직임을 상상

22개의 Ag/AgCl 전극을 사용 EEG
3개의 EOG