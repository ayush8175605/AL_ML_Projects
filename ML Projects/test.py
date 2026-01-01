def greedy_search_decoder(predict_fn, start_token: str, max_len: int) :

      
    """
    Args:
        predict_fn: A function that takes a list of tokens and returns a dictionary
                    of next_token -> log_probability.
        start_token: The initial token.
        max_len: Total number of tokens (including start_token) in final sequences.

    Returns:
        A list of (token_sequence, total_log_prob) tuples, sorted by log_prob descending.
    """
    for i in range(max_len-1):
        scores=predict_fn(start_token)
        max_score_word = max(scores, key=scores.get)
        start_token = start_token+" " +max_score_word

    print(start_token)

        
    
    

def dummy_predict_fn(sequence):
    vocab = ['dog', 'runs', 'fast', 'the', 'a', 'human', 'nice', 'happy', 'raining', 'park']
    import random
    return {token: random.uniform(-1.5, -0.1) for token in vocab}


result = greedy_search_decoder(dummy_predict_fn, "the", 4)

dummy_predict_fn