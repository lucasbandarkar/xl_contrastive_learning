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
    "hindi": lambda examples, question, output_type: (
        "जितना संभव हो सके संक्षिप्त में प्रश्नों के उत्तर दें।\n"
        + "".join(
            f"प्र: {example['question']} केवल एक {example['output_type']} दें\n"
            f"जवाब: {example['prompt_answer']}\n"
            for example in examples
        )
        + f"प्र: {question} केवल एक {output_type} दें\nजवाब:"
    ),
    "thai": lambda examples, question, output_type: (
        "ตอบคำถามต่อไปนี้ให้กระชับที่สุดเท่าที่จะเป็นไปได้\n"
        + "".join(
            f"คำถาม: {example['question']} โปรดตอบเฉพาะ {example['output_type']}\n"
            f"คำตอบ: {example['prompt_answer']}\n"
            for example in examples
        )
        + f"คำถาม: {question} โปรดตอบเฉพาะ {output_type}\nคำตอบ:"
    ),
    "indonesian": lambda examples, question, output_type: (
        "Jawab pertanyaan berikut sesingkat mungkin.\n"
        + "".join(
            f"T: {example['question']} Berikan hanya {example['output_type']}.\n"
            f"J: {example['prompt_answer']}\n"
            for example in examples
        )
        + f"T: {question} Berikan hanya {output_type}.\nJ:"
    ),
    
}
