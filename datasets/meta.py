
DEVICE_INFOS = {
    'OULU': {'spoof': ['_2$', '_3$', '_4$', '_5$'], 'live': ['_1$']},
    'oulu': {'spoof': ['_2$', '_3$', '_4$', '_5$'], 'live': ['_1$']},
    'CASIA_MFSD': {'spoof': ['NM_3', 'NM_4', 'NM_5', 'NM_6', 'NM_7', 'NM_8', 'HR_2', 'HR_3', 'HR_4'], 'live': ['NM_1', 'NM_2', 'HR_1']},
    'casia': {'spoof': ['NM_3', 'NM_4', 'NM_5', 'NM_6', 'NM_7', 'NM_8', 'HR_2', 'HR_3', 'HR_4'], 'live': ['NM_1', 'NM_2', 'HR_1']},
    'Replay_attack': {'spoof': ['attack_highdef', 'attack_print', 'attack_mobile'], 'live' : ['webcam']},
    'replay': {'spoof': ['attack_highdef', 'attack_print', 'attack_mobile'], 'live' : ['webcam']},
    'MSU_MFSD': {'spoof': ['laptop_SD_ipad', 'laptop_SD_iphone', 'laptop_SD_printed', 'android_SD_ipad', 'android_SD_iphone', 'android_SD_printed'],
                 'live' : ['laptop_SD', 'android_SD']},
    'msu': {
        'spoof': ['laptop_SD_ipad', 'laptop_SD_iphone', 'laptop_SD_printed', 'android_SD_ipad', 'android_SD_iphone',
                  'android_SD_printed'],
        'live': ['laptop_SD', 'android_SD']}
}

DOMAIN_ALIAS = {
    'OULU': 'OULU',
    'CASIA_MFSD': 'CASIA',
    'Replay_attack': 'Replay',
    'MSU_MFSD': 'MSU',
}

COLOR_MAP = {
    # blue
    'attack_highdef': "#90E0EF",
    'attack_print': "#CAF0F8",
    'attack_mobile': "#00B4D8",
    'webcam': "#000040",

    # green
    '_2$': "#B7EFC5",
    '_3$': "#92E6A7",
    '_4$': "#6EDE8A",
    '_5$': "#4AD66D",
    '_1$': "#10451D",

    # brown
    'laptop_SD_ipad': "#EDC4B3",
    'laptop_SD_iphone': "#E6B8A2",
    'laptop_SD_printed': "#DEAB90",
    'android_SD_ipad': "#D69F7E",
    'android_SD_iphone': "#CD9777",
    'android_SD_printed': "#C38E70",
    'laptop_SD': "#270007",
    'android_SD': "#421500",

    # red/yellow
    'NM_3': "#F7CAD0",
    'NM_4': "#F9BEC7",
    'NM_5': "#FBB1BD",
    'NM_6': "#FF99AC",
    'NM_7': "#FF7096",
    'NM_8': "#fc6462",
    'HR_2': "#ffe169",
    'HR_3': "#fad643",
    'HR_4': "#edc531",
    'NM_1': "#950104",
    'NM_2': "#8D0000",
    'HR_1': "#704b00",

    'MSU_MFSD': ["#EDC4B3", "#421500",],
    'CASIA_MFSD': ["#ffe169", "#704b00",],
    'Replay_attack': ["#90E0EF", "#000040",],
    'OULU': ["#B7EFC5", "#10451D",],

}

COLOR_MAP_DOMAIN = {
    # 'MSU_MFSD': "#f6511d",
    # 'CASIA_MFSD': "#ffb400",
    # 'Replay_attack': "#00a6ed",
    # 'OULU': "#7fb800"
    'MSU_MFSD': "#6AA84F",
    'CASIA_MFSD': "#3E76D8",
    'Replay_attack': "#D3BD5A",
    'OULU': "#FD8930",
}
