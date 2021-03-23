import ir_datasets
import sources

def test_canard_dev():
    # Just check whether the canard/test can be access and is read correctly
    dataset = ir_datasets.load('canard/dev')
    SAMPLE_SIZE = 3
    i = 0
    queries = []
    for query in dataset.queries_iter():
        queries.append(query)
        i += 1
        if len(queries) >= SAMPLE_SIZE:
            break

    # Check values
    assert queries[0].id == '1'
    assert queries[1].id == '2'
    assert queries[2].id == '3'

    assert queries[0].question == 'What group disbanded?'
    assert queries[1].question == 'When did they disband?'
    assert queries[2].question == 'What kind of music did they play?'

    assert queries[0].answer == 'What group disbanded?'
    assert queries[1].answer == 'When did Zappa and the Mothers of Invention disband?'
    assert queries[2].answer == 'What kind of music did Zappa and the Mothers of Invention play?'


def test_canard_test():
    # Just check whether the canard/test can be access and is read correctly
    dataset = ir_datasets.load('canard/test')
    SAMPLE_SIZE = 3
    i = 0
    queries = []
    for query in dataset.queries_iter():
        queries.append(query)
        i += 1
        if len(queries) >= SAMPLE_SIZE:
            break

    # Check values
    assert queries[0].id == '1'
    assert queries[1].id == '2'
    assert queries[2].id == '3'

    assert queries[0].question == 'Did they have any clues?'
    assert queries[1].question == 'How did they target her email?'
    assert queries[2].question == 'Did they get into trouble for that?'

    assert queries[0].answer == 'Did investigators have any clues in the unresolved murder of Anna Politkovskaya?'
    assert queries[1].answer == 'How did FSB target the murdered journalist Anna Politkovskaya\'s email?'
    assert queries[2].answer == 'Did FSB get into trouble for attacking the email account of Anna Politkovskaya\'s?'