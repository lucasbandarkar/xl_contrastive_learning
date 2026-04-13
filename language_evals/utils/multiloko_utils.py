MULTILOKO_DATA_ROOT = "benchmark_data"

MULTILOKO_PROMPT_BUILDERS = {
    "english": lambda examples, question, output_type: (
        "Answer the given question as concisely as possible.\n"
        + "".join(
            (
                f"Q: {example['question']} Produce only a {example['output_type']}. "
                f"A: {example['prompt_answer']}\n"
            )
            for example in examples
        )
        + f"Q: {question} Produce only a {output_type}. A:"
    ),
    "arabic": lambda examples, question, output_type: (
        "يرجى الإجابة عن السؤال التالي بإيجاز قدر الإمكان.\n"
        + "".join(
            (
                "يجب أن تكون الإجابة في شكل "
                f"{example['output_type']}\n"
                f"السؤال: {example['question']}\n"
                f"الإجابة: {example['prompt_answer']}\n"
            )
            for example in examples
        )
        + (
            "يرجى الإجابة عن السؤال التالي بإيجاز قدر الإمكان. "
            f"يجب أن تكون الإجابة في شكل {output_type}.\n"
            f"السؤال: {question}\n"
            "الإجابة:"
        )
    ),
    "bengali": lambda examples, question, output_type: (
        "যতটা সম্ভব সংক্ষিপ্তভাবে প্রশ্নগুলোর উত্তর দাও।\n"
        + "".join(
            f"প্র: {example['question']} শুধু একটি {example['output_type']} দাও\n"
            f"উ: {example['prompt_answer']}\n"
            for example in examples
        )
        + f"প্র: {question} শুধু একটি {output_type} দাও\nউ:"
    ),
    "farsi": lambda examples, question, output_type: (
        "به این سوال تا حد امکان دقیق پاسخ دهید.\n"
        + "".join(
            (
                f"سوال: {example['question']}\n"
                f"فقط با {example['output_type']} پاسخ دهید.\n"
                f"جواب: {example['prompt_answer']}\n"
            )
            for example in examples
        )
        + f"سوال: {question}\nفقط با {output_type} پاسخ دهید.\nجواب:"
    ),
    "french": lambda examples, question, output_type: (
        "Réponds à la question le plus brièvement possible.\n"
        + "".join(
            (
                f"Q : {example['question']}\n"
                f"Ta réponse doit être du type : {example['output_type']}.\n"
                f"R : {example['prompt_answer']}\n"
            )
            for example in examples
        )
        + (
            f"Q : {question}\n"
            f"Ta réponse doit être du type : {output_type}.\n"
            "R :"
        )
    ),
}