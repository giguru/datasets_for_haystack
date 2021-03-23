import ir_datasets
import sources

def test_orconvqa_dev():
    # Just check whether the orconvqa/test can be access and is read correctly
    dataset = ir_datasets.load('orconvqa/dev')
    SAMPLE_SIZE = 3
    i = 0
    queries = []
    for query in dataset.queries_iter():
        queries.append(query)
        i += 1
        if len(queries) >= SAMPLE_SIZE:
            break

    # Check values
    assert queries[0].document_id == 'C_9b44280b1d3a4ee3b42406a86a21c532_1&C_b2fe1b225c6044d0b480f18deb33ff0d_1@0'
    assert queries[1].document_id == 'C_9b44280b1d3a4ee3b42406a86a21c532_1&C_b2fe1b225c6044d0b480f18deb33ff0d_1@0'
    assert queries[2].document_id == 'C_9b44280b1d3a4ee3b42406a86a21c532_1&C_b2fe1b225c6044d0b480f18deb33ff0d_1@0'

    assert queries[0].question == 'Where was Michael Bennett born?'
    assert queries[1].question == 'When  was Michael Bennett  born?'
    assert queries[2].question == 'Who are Michael Bennett\'s parents?'

    assert queries[0].answer == 'Bennett was born Michael Bennett DiFiglia in Buffalo, New York,'
    assert queries[0].no_answer == False
    assert queries[1].answer == 'CANNOTANSWER'
    assert queries[1].no_answer == True
    assert queries[2].answer == 'the son of Helen (nee Ternoff), a secretary, and Salvatore Joseph DiFiglia, a factory worker.'
    assert queries[2].no_answer == False


def test_orconvqa_test():
    # Just check whether the orconvqa/test can be access and is read correctly
    dataset = ir_datasets.load('orconvqa/test')
    SAMPLE_SIZE = 3
    i = 0
    queries = []
    for query in dataset.queries_iter():
        queries.append(query)
        i += 1
        if len(queries) >= SAMPLE_SIZE:
            break

    # Check values
    assert queries[0].document_id == 'C_0aaa843df0bd467b96e5a496fc0b033d_1@0'
    assert queries[1].document_id == 'C_0aaa843df0bd467b96e5a496fc0b033d_1@1'
    assert queries[2].document_id == 'C_0aaa843df0bd467b96e5a496fc0b033d_1@0'

    assert queries[0].question == 'Did investigators have any clues in the unresolved murder of Anna Politkovskaya?'
    assert queries[1].question == 'How did FSB target the murdered journalist Anna Politkovskaya\'s email?'
    assert queries[2].question == 'Did FSB get into trouble for attacking the email account of Anna Politkovskaya\'s?'

    assert queries[0].answer == 'probably FSB) are known to have targeted the webmail account of the murdered Russian journalist Anna Politkovskaya.'
    assert queries[0].no_answer == False
    assert queries[1].answer == 'On 5 December 2005, RFIS initiated an attack against the account annapolitovskaya@US Provider1, by deploying malicious software'
    assert queries[1].no_answer == False
    assert queries[2].answer == 'CANNOTANSWER'
    assert queries[2].no_answer == True

test_orconvqa_test()