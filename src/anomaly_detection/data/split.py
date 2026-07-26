from ..config import TRAIN_SET_SIZE
from ..config import TEST_SIZE
from ..config import VAL_SET_SIZE


def find_parasitized_count(dataset):
    count = 0
    for cell in dataset:
        if(cell['label'] == 0):
            count +=1
    return count

def split(malaria_builder, batch_size):
    malaria_dataset = malaria_builder.as_dataset(split="train")
    if batch_size == 0:
        batch_size = len(malaria_dataset)

    malaria_dataset = malaria_dataset.shuffle(buffer_size = batch_size , reshuffle_each_iteration=False)
    total_size = len(malaria_dataset)
    if(total_size > batch_size):
        total_size = batch_size

    TEST_DATA = malaria_dataset.take(int(total_size * TEST_SIZE))
    TRAINING_DATA = malaria_dataset.skip(len(TEST_DATA)).take(int(total_size * TRAIN_SET_SIZE))
    VAL_DATA = malaria_dataset.skip(len(TEST_DATA)).skip(len(TRAINING_DATA)).take(int(total_size * VAL_SET_SIZE))

    print(f'Training data : {len(TRAINING_DATA)} , Validation Data : {len(VAL_DATA)}, Test Data: {len(TEST_DATA)}')

    parasitized_count = find_parasitized_count(TEST_DATA)
    print(f'parasitized count {int((parasitized_count/len(TEST_DATA))*100) } for Test data')


    parasitized_count = find_parasitized_count(TRAINING_DATA)
    print(f'parasitized count {int((parasitized_count/len(TRAINING_DATA))*100) } for training data')


    parasitized_count = find_parasitized_count(VAL_DATA)
    print(f'parasitized percentage {int((parasitized_count/len(VAL_DATA))*100) } for validation data')

    return TRAINING_DATA, VAL_DATA, TEST_DATA
