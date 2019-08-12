#
# epitome and defcom tuples
#

EPITOME_AND_DEFCOM_CELLS = [('K562', 'k562'),
                ('GM12878', 'gm12878'),
                ('HepG2', 'hepg2'),
                ('H1-hESC', 'h1hesc')]

EPITOME_AND_DEFCOM_TFS = [
    ('CEBPB', 'cebpb'),
    ('CHD2', 'chd2'),
    ('CTCF', 'ctcf'),
    #('ep300', 'EP300'), # doesn't exist in epitome, therefore we exclude it
    #('gabpa', ''), # doesn't exist in epitome, therefore we exclude it
    #('rest', ''),   # doesn't exist in epitome, therefore we exclude it
    ('JunD', 'jund'),
    ('MafK', 'mafk'),
    ('Max', 'max'),
    ('c-Myc', 'myc'),
    ('Nrf1', 'nrf1'),
    ('Rad21', 'rad21'),
    ('RFX5', 'rfx5'),
    ('SRF', 'srf'),
    ('TAF1', 'taf1'),
    ('TBP', 'tbp'),
    ('USF2', 'usf2')]

#
# defcom cells and tfs
#

DEFCOM_CELLS = ['k562', 'gm12878', 'hepg2', 'h1hesc']
DEFCOM_TFS = ['cebpb', 'chd2', 'ctcf', 'ep300', 'gabpa', 
       'jund', 'mafk', 'max', 'myc', 'nrf1', 'rad21', 
       'rest', 'rfx5', 'srf', 'sp1', 'taf1', 'tbp', 'usf2']

#
# epitome cells and tfs
#

EPITOME_CELLS = ['K562', 'GM12878', 'HepG2', 'H1-hESC']
EPITOME_TFS = ['CEBPB', 'CHD2', 'CTCF', 'JunD', 'MafK', 'Max', 'c-Myc', 
               'Nrf1', 'Rad21', 'RFX5', 'SRF', 'TAF1', 'TBP', 'USF2' 
]