**Context**:
    We use a machine learning model to determine whether ChatGPT has generated a piece of source code. The classification engine is based on GraphCodeBERT. The model we used is part of the GPTSniffer (https://github.com/MDEGroup/GPTSniffer).
    The model yields the following probability scores, marked as **Classification Results**, for the target source code, marked as **Target Source Code**, below. Moreover, along with the probability scores for the target source code, you are provided with some samples of the ChatGPT-generated and human-written code (marked as **Examples**) that were classified by the same model. 

    **Examples**:
    - ChatGPT-generated samples:
        {related_examples_chatGPT}

    - Human-written samples:
        {related_examples_human}

    **Target Source Code**:
    {source_code}

    **Classification Results**:
        - Probability of being ChatGPT-generated: {probability_chatgpt}%.
        - Probability of being Human-written: {probability_human}%.

    **Instruction**:
    ญlease provide a comprehensive explanation and reasoning why the model predicts such code as human-written or AI-generated containing the following sections:

    **Answer structure**:
    
    **Overview**
    - Briefly summarize the most influential features or code excerpts that led to the classification as **{more_likely_classification}**. This must not exceed 20 lines.
    
    End the overview with a markdown horizontal line (e.g., `---`).

    **Highlight and Explain Key Code Lines**:
    - Key Code Lines: Extract five code lines or statements from the target source code that demonstrate {more_likely_classification} characteristics.
        - Explain Each Code Line:
            - Describe why the code lines indicates {more_likely_classification} patterns.
            - Detail the features or patterns present in the code that lead to {more_likely_classification}.
            

    **Comparison to Previous Examples**:
    - Analyze how the features in the target source code align with or differ from the given code examples and their classifications.
    - Compare these features with similar patterns found in the dataset examples. For each comparison, provide the corresponding code snippet from the example alongside the target code excerpt. This side-by-side presentation will help readers understand the model's classification better.
    - When referring to a specific example (e.g., "Example 1"), include its relevant code snippet for a direct visual comparison with the target source code.
    - Discuss how these similarities or differences support the classification decision.

    Your final explanation should be detailed, structured and is in a markdown format. 