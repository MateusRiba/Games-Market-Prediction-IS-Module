import numpy as np

sentiment_dict = {'Universal acclaim': 5,
'Generally favorable': 4,
'Mixed or average': 3,
'Generally unfavorable': 2,
'Overwhelming dislike': 1}


rating_dict = {
    'RP': 0,    # Rating Pending - não classificado
    'E': 1,     # Everyone
    'K-A': 2,   # Kids to Adults (antigo, alinhado com E10+)
    'E10+': 2,  # Everyone 10+
    'T': 3,     # Teen
    'M': 4,     # Mature
    'AO': 5     # Adults Only
}


broad_platform_mapping = {
    'nintendo_64': 'nintendo_console', 'wii': 'nintendo_console', 'gamecube': 'nintendo_console',
    'wii_u': 'nintendo_console', 'nintendo_switch': 'nintendo_console', 'nes': 'nintendo_console',
    'snes': 'nintendo_console', 'game_boy_color': 'nintendo_console', 'nintendo_3ds': 'nintendo_console',
    'nintendo_ds': 'nintendo_console', 'game_boy': 'nintendo_console',

    'playstation': 'playstation_console', 'playstation_2': 'playstation_console',
    'playstation_3': 'playstation_console', 'playstation_4': 'playstation_console',
    'playstation_5': 'playstation_console',

    'xbox': 'xbox_console', 'xbox_360': 'xbox_console', 'xbox_one': 'xbox_console',
    'xbox_series_x': 'xbox_console',

    'pc': 'pc',

    'ds': 'portable', '3ds': 'portable', 'game_boy_advance': 'portable',
    'psp': 'portable', 'playstation_vita': 'portable',
    'ios_ios_iphone_ipad': 'portable', 'android': 'portable',

    'dreamcast': 'other_consoles',
    'meta_quest': 'other_consoles'

}


developer_name_mapping = {
    # --- GRUPOS GIGANTES (Empresas Mãe / Ecossistemas) ---
    'nintendo': 'nintendo', # Mantém o nome principal
    'nintendo ead tokyo': 'nintendo', 'nintendo r&d1': 'nintendo', 'nintendo spd': 'nintendo',
    'nintendo ead': 'nintendo', 'intelligent systems': 'nintendo', 'hal labs': 'nintendo',
    'game freak': 'nintendo', # Pokémon
    'camelot software planning': 'nintendo', # Mario Golf/Tennis, Golden Sun
    'nd cube': 'nintendo', # Mario Party

    'rockstar games': 'rockstar games', 'rockstar north': 'rockstar games', 'rockstar san diego': 'rockstar games',
    'rockstar vancouver': 'rockstar games', 'rockstar toronto': 'rockstar games', 'rockstar london': 'rockstar games',
    'rockstar new england': 'rockstar games', 'rockstar leeds': 'rockstar games',

    'electronic arts': 'electronic arts', 'ea canada': 'electronic arts', 'ea sports': 'electronic arts',
    'ea dice': 'electronic arts', 'ea la': 'electronic arts', 'bioware': 'electronic arts',
    'ea tiburon': 'electronic arts', 'criterion games': 'electronic arts', # Need for Speed
    'pandemic studios': 'electronic arts', # Star Wars Battlefront
    'visceral games': 'electronic arts', # Dead Space

    'ubisoft': 'ubisoft', 'ubisoft montreal': 'ubisoft', 'ubisoft paris': 'ubisoft',
    'red storm entertainment': 'ubisoft', # Rainbow Six, Ghost Recon
    'ubisoft montpellier': 'ubisoft', # Rayman

    'konami': 'konami', 'konami computer entertainment japan': 'konami', 'kcet': 'konami',

    'square enix': 'square enix', 'square soft': 'square enix', 'enix': 'square enix',

    'sony interactive entertainment': 'sony interactive entertainment',
    'sce japan studio': 'sony interactive entertainment', 'sony online entertainment': 'sony interactive entertainment',
    '989 sports': 'sony interactive entertainment', # Mapeia para Sony
    'scee london studio': 'sony interactive entertainment', # Adicionado
    'scea san diego studios': 'sony interactive entertainment', # Adicionado
    'media molecule': 'sony interactive entertainment', # LittleBigPlanet, Dreams
    'insomniac games': 'sony interactive entertainment', # Ratchet & Clank, Spider-Man
    'guerrilla': 'sony interactive entertainment', # Killzone, Horizon Zero Dawn
    'housemarque': 'sony interactive entertainment', # Returnal
    'naughty dog': 'sony interactive entertainment', # Uncharted, The Last of Us

    'microsoft': 'microsoft game studios', # Nome canônico da empresa
    'microsoft game studios': 'microsoft game studios', # Mapeia
    'lionhead studios': 'microsoft game studios', # Fable
    'rare ltd': 'microsoft game studios', # Mapeia (foi comprada pela Microsoft/Xbox)

    'activision': 'activision blizzard', # Agrupando em uma controladora maior
    'blizzard entertainment': 'activision blizzard', # Agrupando
    'treyarch': 'activision blizzard', # Call of Duty
    'vicarious visions': 'activision blizzard', # Crash Bandicoot N. Sane Trilogy
    'infinity ward': 'activision blizzard', # Call of Duty
    'sledgehammer games': 'activision blizzard', # Call of Duty
    'raven software': 'activision blizzard', # Call of Duty

    'take-two interactive': 'take-two interactive', # Empresa-mãe
    '2k games': 'take-two interactive', # Publicadora da Take-Two
    'firaxis games': 'take-two interactive', # Civilizati
    'visual concepts': 'take-two interactive', # NBA 2K

    'valve software': 'valve software', # Dota 2, Half-Life, Portal

    # --- DESENVOLVEDORES INDEPENDENTES / OUTRAS GRANDES / NICHOS ---
    'capcom': 'capcom',
    'telltale games': 'telltale games',
    'omega force': 'omega force',
    'tose': 'tose',
    'maxis': 'maxis', # The Sims
    'digital eclipse': 'digital eclipse',
    'sega': 'sega',
    'nippon ichi software': 'nippon ichi software',
    'namco': 'namco',
    'atlus': 'atlus', # Persona
    'bandai namco games': 'bandai namco games',
    'fromsoftware': 'fromsoftware', # Dark Souls
    'arc system works': 'arc system works', # Guilty Gear
    'milestone srl': 'milestone srl',
    'team17': 'team17', # Worms
    'midway': 'midway', # Mortal Kombat
    'creative assembly': 'creative assembly', # Total War
    'frontier developments': 'frontier developments', # Elite Dangerous
    'platinumgames': 'platinumgames', # Bayonetta
    'ryu ga gotoku studios': 'sega', # Yakuza
    'snk playmore': 'snk playmore', # King of Fighters
    'eurocom': 'eurocom',
    'travellers tales': 'travellers tales', # LEGO Games
    'acquire': 'acquire',
    'sumo digital': 'sumo digital',
    'koei tecmo games': 'koei tecmo games', # A controladora
    'tecmo': 'koei tecmo games', # Agrupando Tecmo
    'koei': 'koei tecmo games', # Agrupando Koei
    'paradox development studio': 'paradox development studio', # Jogos de estratégia
    'gameloft': 'gameloft', # Mobile
    'volition inc': 'volition inc', # Saints Row
    'lucasfilm games': 'lucasfilm games', # Antiga LucasArts
    'artificial mind and movement': 'artificial mind and movement',
    'obsidian entertainment': 'obsidian entertainment', # RPGs
    'arika': 'arika',
    'hudson': 'hudson', # Bomberman
    'relic entertainment': 'relic entertainment', # Company of Heroes
    '3do': '3do',
    'monolith productions': 'monolith productions', # Middle-earth
    'now production': 'now production',
    'techland': 'techland', # Dying Light
    'double fine productions': 'double fine productions', # Psychonauts
    'firaxis games': 'firaxis games', # Civilization
    'inti creates': 'inti creates',
    'supermassive games': 'supermassive games', # The Dark Pictures Anthology
    'game arts': 'game arts',
    'grasshopper manufacture': 'grasshopper manufacture',
    'sports interactive': 'sports interactive', # Football Manager
    'treasure': 'treasure',
    'blitz games': 'blitz games',
    'dontnod entertainment': 'dontnod entertainment',
    'racjin': 'racjin',
    'naughty dog': 'naughty dog',
    'krome studios': 'krome studios',
    'eighting': 'eighting',
    'rainbow studios': 'rainbow studios',
    'cyanide': 'cyanide',
    'jupiter corporation': 'jupiter corporation',
    'tamsoft': 'tamsoft',
    'q-games': 'q-games',
    'bethesda game studios': 'bethesda game studios',
    'id software': 'bethesda game studios', # Doom, Quake
    'arkane studios': 'bethesda game studios', # Dishonored
    'bungie': 'bungie', # Destiny
    'shinen': 'shinen',
    'popcap': 'popcap', # Plants vs Zombies
    'milestone srl': 'milestone srl',

    # --- ARTEFATOS/TERMOS GENÉRICOS QUE DEVEM SER MAPEADOS PARA NaN ---
    'inc': np.nan, # Mapeia 'inc' para NaN
    'ltd': np.nan, # Mapeia 'ltd' para NaN
    'hand inc': np.nan, # Mapeia 'hand inc' para NaN
    'gmbh': np.nan, # Se aparecer
    's.a.': np.nan, # Se aparecer
    'srl': np.nan, # Se aparecer
    'co': np.nan, # Se aparecer
    'corp': np.nan, # Se aparecer
    'corporation': np.nan, # Se aparecer
    'games': np.nan, # Se aparecer como termo isolado

    # Continue adicionando mais mapeamentos se você encontrar outras variações
    # que o seu 'unmapped_frequent_devs' mostrar após esta execução.
}

important_developers = [
    'nintendo', 'nintendo ead tokyo', 'rockstar games', 'valve', 'cd projekt red', 'naughty dog',
    'blizzard entertainment', 'sony', 'microsoft', 'electronic arts', 'ubisoft', 'capcom',
    'konami', 'square enix', 'rare ltd.', 'fromsoftware', 'take-two interactive',
    'irrational games', 'media molecule', 'hal labs', 'intelligent systems', 'telltale games',
    'omega force', 'tose', 'maxis', 'digital eclipse', 'sega', 'nippon ichi software',
    'visual concepts', 'namco', 'atlus', 'bandai namco games', 'milestone srl',
    'team17', 'midway', 'insomniac games', 'creative assembly', 'frontier developments',
    'platinumgames', 'snk playmore', 'eurocom', 'travellers tales', 'acquire', 'sumo digital',
    'koei tecmo games', 'paradox development studio', 'gameloft', 'volition inc',
    'lucasfilm games', 'obsidian entertainment', 'arika', 'hudson', 'relic entertainment',
    '3do', 'monolith productions', 'now production', 'techland', 'double fine productions',
    'firaxis games', 'inti creates', 'supermassive games', 'game arts', 'grasshopper manufacture',
    'sports interactive', 'treasure', 'blitz games', 'dontnod entertainment', 'racjin',
    'krome studios', 'eighting', 'rainbow studios', 'cyanide', 'jupiter corporation', 'tamsoft',
    'q-games', 'bethesda game studios', 'id software', 'arkane studios', 'bungie', 'shinen',
    'popcap', 'bandai namco games', 'activision', 'sony interactive entertainment', 'koei tecmo games',
    'lucasfilm games', 'microsoft', 'rare ltd.', 'electronic arts', 'ubisoft', 'konami', 'square enix',
    'fromsoftware', 'arc system works', 'milestone srl', 'level 5', 'blizzard entertainment', 'dimps corporation',
    'falcom', 'genki', 'idea factory', 'avalanche software', 'tt games', 'saber interactive',
    'creat studios', 'squaresoft', 'haemimont games', 'housemarque', 'infinity ward',
    'griptonite games', 'radical entertainment', 'ubisoft montpellier', 'kcej', 'monolith soft',
    'frogwares', 'guerrilla', 'eutechnyx', 'pandemic studios', 'nival interactive', 'nd cube',
    'stainless games', 'terminal reality', 'bizarre creations', '10tons', 'mediavision',
    'hudson soft', 'monster games inc', 'indieszero', 'io interactive', 'kojima productions',
    'neverland', 'natsume', 'iron galaxy studios', 'arkane studios', 'epic games', 'mass media',
    'ryu ga gotoku studios', 'activision blizzard', # Nomes canônicos mais amplos
]


publisher_name_mapping = {
    # --- GRANDES CONGLOMERADOS / FAMÍLIAS DE PUBLISHERS (Amplo Agrupamento Semântico) ---

    # Nintendo
    'nintendo': 'nintendo', 'nintendo of america': 'nintendo', 'nintendo of europe': 'nintendo',
    'nintendo of australia': 'nintendo', 'nintendo co ltd': 'nintendo', 'i que': 'nintendo', # iQue é joint venture da Nintendo na China
    'the pokemon company': 'nintendo', # Associado primariamente à Nintendo (franquia Pokémon)

    # Sony Interactive Entertainment (PlayStation)
    'sony interactive entertainment': 'sony interactive entertainment', 'scea': 'sony interactive entertainment', # Sony Computer Entertainment America
    'scei': 'sony interactive entertainment', 'scee': 'sony interactive entertainment', # Europe
    'sce australia': 'sony interactive entertainment', 'sony computer entertainment': 'sony interactive entertainment',
    'playstation mobile': 'sony interactive entertainment', # Serviços mobile
    '989 sports': 'sony interactive entertainment', # Estúdio antigo da Sony

    # Microsoft (Xbox Game Studios)
    'microsoft': 'microsoft', 'microsoft game studios': 'microsoft', 'xbox game studios': 'microsoft',
    'microsoft studios': 'microsoft', 'rare ltd': 'microsoft', # Rare Ltd. (desenvolvedora) foi comprada pela Microsoft/Xbox
    'bungie': 'microsoft', # Publicaram Halo inicialmente para Microsoft (depois se tornaram multi-plataforma e foram para Sony) - Decisão: mapear para Microsoft pelo histórico principal no contexto de Xbox.

    # Electronic Arts (EA)
    'electronic arts': 'electronic arts', 'ea': 'electronic arts', 'ea sports': 'electronic arts',
    'ea canada': 'electronic arts', 'ea dice': 'electronic arts', 'ea la': 'electronic arts',
    'maxis': 'electronic arts', # The Sims
    'bioware': 'electronic arts', # Mass Effect, Dragon Age
    'criterion games': 'electronic arts', # Need for Speed
    'pandemic studios': 'electronic arts', # Star Wars Battlefront
    'popcap': 'electronic arts', # Plants vs Zombies

    # Ubisoft
    'ubisoft': 'ubisoft', 'ubisoft entertainment': 'ubisoft', 'ubisoft montreal': 'ubisoft', 'ubisoft paris': 'ubisoft',
    'ubisoft massive': 'ubisoft', 'red storm entertainment': 'ubisoft',

    # Activision Blizzard
    'activision blizzard': 'activision blizzard', 'activision': 'activision blizzard', 'blizzard entertainment': 'activision blizzard',
    'treyarch': 'activision blizzard', 'vicarious visions': 'activision blizzard', 'infinity ward': 'activision blizzard',
    'sledgehammer games': 'activision blizzard', 'raven software': 'activision blizzard',

    # Take-Two Interactive
    'take-two interactive': 'take-two interactive', '2k games': 'take-two interactive', 'rockstar games': 'take-two interactive',
    'private division': 'take-two interactive', # Selo indie da Take-Two
    'visual concepts': 'take-two interactive', # NBA 2K
    'firaxis games': 'take-two interactive', # Civilization

    # Capcom
    'capcom': 'capcom', 'capcom usa': 'capcom', 'capcom co ltd': 'capcom',

    # Konami
    'konami': 'konami', 'konami computer entertainment japan': 'konami', 'kcet': 'konami',

    # Square Enix
    'square enix': 'square enix', 'square': 'square enix', 'enix': 'square enix', 'taito corporation': 'square enix', # Taito é subsidiária

    # Bandai Namco Entertainment
    'bandai namco entertainment': 'bandai namco entertainment', 'bandai namco games': 'bandai namco entertainment',
    'namco': 'bandai namco entertainment', 'bandai': 'bandai namco entertainment',

    # SEGA
    'sega': 'sega', 'sega of america': 'sega', 'sega europe': 'sega', 'sonic team': 'sega',
    'ryu ga gotoku studios': 'sega', # Estúdios Yakuza

    # Bethesda Softworks (ZeniMax Media)
    'bethesda softworks': 'bethesda softworks', 'zenimax media': 'bethesda softworks',
    'bethesda game studios': 'bethesda softworks', 'id software': 'bethesda softworks', 'arkane studios': 'bethesda softworks',

    # --- OUTRAS PUBLICADORAS IMPORTANTES / CONSOLIDADAS / NICHOS ---
    'valve software': 'valve software', # Dota 2, Half-Life, Steam
    'atlus': 'atlus', # Persona
    'telltale games': 'telltale games',
    'thq': 'thq', # Se for THQ Nordic, mapear especificamente se não for coberto
    'deep silver': 'deep silver',
    'focus home interactive': 'focus home interactive',
    'paradox interactive': 'paradox interactive', # Jogos de estratégia
    '505 games': '505 games',
    'koei tecmo games': 'koei tecmo games', 'koei': 'koei tecmo games', 'tecmo': 'koei tecmo games', # Consolidando
    'annapurna interactive': 'annapurna interactive', # Indiel
    'humble bundle': 'humble bundle',
    'devolver digital': 'devolver digital',
    'team17': 'team17', # Worms
    'tinybuild': 'tinybuild',
    'raw fury': 'raw fury',
    'arc system works': 'arc system works', # Jogos de luta
    'chucklefish': 'chucklefish',
    'tripwire interactive': 'tripwire interactive',
    'merge games': 'merge games',
    'daedalic entertainment': 'daedalic entertainment',
    'rebellion': 'rebellion', # Sniper Elite
    'spike chunsoft': 'spike chunsoft',
    'nicalis': 'nicalis',
    'soedesco': 'soedesco',
    'maximum games': 'maximum games',
    'modus games': 'modus games',
    'xseed games': 'xseed games',
    'digerati': 'digerati',
    'dotemu': 'dotemu', # Arcade
    'plug in digital': 'plug in digital',
    'gamera interactive': 'gamera interactive',
    'pm studios': 'pm studios',
    'soedesco': 'soedesco',

    # --- PUBLISHERS QUE PODEM SER LOCAIS OU MENORES COM POUCAS OCORRÊNCIAS ---
    'gradiente': 'other_publisher', # Exemplo que apareceu nos seus dados
    'success': 'other_publisher', # Exemplo que apareceu nos seus dados
    'i n c': 'other_publisher', # Tratamento para 'inc' que pode ter sido limpo assim (se não for np.nan)
    'nexon': 'nexon', # Publisher de MMOs
    'kakao games': 'kakao games', # Publisher Coreano
    'my.games': 'my.games', # Publisher Russo
    'perfect world entertainment': 'perfect world entertainment', # Publisher de MMOs

    # --- ARTEFATOS / TERMOS GENÉRICOS QUE DEVEM SER MAPEADOS PARA NaN ---
    'inc': np.nan, 'ltd': np.nan, 'gmbh': np.nan, 's.a.': np.nan, 'srl': np.nan,
    'co': np.nan, 'corp': np.nan, 'corporation': np.nan, 'games': np.nan,
    'entertainment': np.nan, 'interactive': np.nan, 'digital': np.nan,
    'systems': np.nan, 'publishing': np.nan, 'studios': np.nan, 'llc': np.nan,
    'group': np.nan, 'media': np.nan, 'software': np.nan, 'global': np.nan,
    'solutions': np.nan, 'interactive': np.nan, 'holding': np.nan, 'development': np.nan,
    'of america': np.nan, 'of europe': np.nan, 'of australia': np.nan, # Removendo os "of X"
    'uk': np.nan, 'jp': np.nan, 'eu': np.nan, 'us': np.nan, 'usa': np.nan, # Abreviações de região
    'co': np.nan, # Se aparecer como termo isolado
    'international': np.nan, 'company': np.nan, 'l p': np.nan, # Termos comuns
    'inc': np.nan, # Se aparecer como termo isolado


    # Adicione mais mapeamentos aqui conforme você identificar mais variações
    # Lembre-se: chaves são os nomes LIMPOS e padronizados que aparecem no seu df_publisher_features['publisher_cleaned_list'],
    # valores são os NOMES CANÔNICOS que você quer que eles se tornem.
}

important_publishers = [
    'nintendo', 'electronic arts', 'ubisoft', 'activision blizzard', 'take-two interactive',
    'sony interactive entertainment', 'microsoft', 'capcom', 'konami', 'square enix',
    'bandai namco entertainment', 'sega', 'valve software', 'bethesda softworks', 'cd projekt',
    'atlus', 'telltale games', 'thq', 'deep silver', 'focus home interactive', '505 games',
    'koei tecmo games', 'annapurna interactive', 'humble bundle', 'devolver digital',
    'team17', 'tinybuild', 'raw fury', 'arc system works', 'chucklefish', 'tripwire interactive',
    'merge games', 'daedalic entertainment', 'private division', 'rebellion', 'spike chunsoft',
    'nicalis', 'soedesco', 'maximum games', 'modus games', 'xseed games', 'digerati',
    # Adicione outros NOMES CANÔNICOS importantes aqui que você queira proteger!
]
