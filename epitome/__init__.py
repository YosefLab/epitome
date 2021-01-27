import os
from os.path import expanduser

__path__ = __import__('pkgutil').extend_path(__path__, __name__)

S3_DATA_PATH = 'https://epitome-data.s3-us-west-1.amazonaws.com'

# os env that should be set by user to explicitly set the data path
EPITOME_DATA_PATH_ENV= "EPITOME_DATA_PATH" # Must be an absolute path.
EPITOME_GENOME_ASSEMBLY_ENV = "EPITOME_GENOME_ASSEMBLY"

# data files required by epitome
POSITIONS_FILE = "all.pos.bed.gz"
FEATURE_NAME_FILE = "feature_name"
REQUIRED_FILES = [POSITIONS_FILE,"train.npz","valid.npz", FEATURE_NAME_FILE,"test.npz"]
EPITOME_GENOME_ASSEMBLIES = ['hg19']

def GET_EPITOME_USER_PATH():
    return os.path.join(os.path.expanduser('~'), '.epitome')

def LIST_GENOMES():
    return ", ".join(EPITOME_GENOME_ASSEMBLIES)

def GET_DATA_PATH():
    """
    Check if user has set env variable that specifies data path.
    Otherwise, use default location.
    
    Returns:
        location of epitome data with all required files
    """
    epitome_data_path  = os.environ.get(EPITOME_DATA_PATH_ENV)
    epitome_assembly = os.environ.get(EPITOME_GENOME_ASSEMBLY_ENV)
    print("HELLO !!!")
    
    # Throw error if both ENV variables are set
    assert((epitome_data_path is not None) and (epitome_assembly is not None)), "Only specify either the %s env variable or %s env variable. Cannot define both." % (EPITOME_DATA_PATH_ENV, EPITOME_GENOME_ASSEMBLY_ENV)
    
    # Return specified data path if env variable is set
    if (epitome_data_path_env is not None):
        return epitome_data_path_env
    else:
        # Default to the hg19 assembly if assembly and data path is not specified
        if (epitome_assembly is None):
            epitome_assembly = 'hg19'
            print("Warning: genome assembly %s env variable was not set. Defaulting genome assembly to %s." % (EPITOME_GENOME_ASSEMBLY_ENV, epitome_assembly))
        epitome_data_dir_path = os.path.join(GET_EPITOME_USER_PATH(), 'data')
        return os.path.join(epitome_data_dir_path, epitome_assembly)
    
    