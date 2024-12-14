from imports import *

print(torch.__version__)
print(torch.cuda.is_available())

app = Flask(__name__, template_folder='../templates')

model_name = "google/flan-t5-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

text2text_generation_pipeline = pipeline("text2text-generation", model=model, tokenizer=tokenizer)

def get_audio_base64(text):
  try:
    tts = gTTS(text=text, lang='en')
    audio_file_path = f"{uuid.uuid4().hex}.mp3"
    tts.save(audio_file_path)

    with open(audio_file_path, 'rb') as f:
            audio_bytes = f.read()
    audio_base64 = b64encode(audio_bytes).decode('ascii')
    os.remove(audio_file_path) # clean up file immediately
    return audio_base64
  except Exception as e:
      print(f"Error generating audio from gTTS: {e}")
      return ""
  except AssertionError as e:
      print(f" Assertion Error from gTTS(Likely empty text): {e}")
      return ""



# Sample FAQs
# faqs = {
#     "When is the proposal deadline": "The deadline of project proposal is on June 2025",
#     "When is the deadline for the project?": "The deadline is mostly annouced by the HOD of the faculty in which you find your self. Always be alert for any information about the dealine mentioned or shared by the HOD for the project documentation.",
#     "What should the project proposal look like": "The project proposal should contain: Your name(s), faculty, department, course, title of the project, student IDs, project idea, and justification of the project idea.",
#     "How do I choose a suitable project topic?": " Select a topic that genuinely interests you, choose a topic that has real-world implications, seek guidance from your supervisors to ensure feasibility and alignment with course requirements, analyze previous successful projects to gain inspiration.",
#     "What is the ideal length for a project proposal?": "Follow specific word count or page limit guidelines provided by your university, avoid unnecessary details, highlight the project's objectives, methodology, and expected outcomes.",
#     "What is the difference between aims and objectives?" : "Aims: Broad goals that outline the overall direction of your project. Objectives: Specific, measurable steps to achieve your aims.",
#     "How do I structure my project documentation?": "Adhere to the recommended structure, including introduction, literature review, methodology, implementation, discussion, and conclusion, use clear headings and subheadings, use appropriate citation styles(e.g., APA, MLA, Chicago, Harvard).",
#     "How do I write a good literature review?": "Use academic databases and reputable websites to find relevant research papers, articles, and books, Evaluate the strengths and weaknesses of each source, Paraphrase and cite sources correctly and avoid plagiarism, Combine the findings of different sources to form a coherent narrative.",
#     "How do I present my findings effectively?": "Employ charts, graphs, and diagrams to enhance understanding, Focus on the most significant results, Explain the implications of your findings, Acknowledge any limitations of your research",
#     "What are the key elements of a good conclusion?": " Summarize the main points of your project, Provide a clear answer to your research question, Explain the broader significance of your findings, Propose potential areas for further investigation",
#     "How many pages does a project documentation document have?": "It is recommended that the text number ranges between 30 to 50 pages for Diploma and 40 to 60 pages for Degree."
# }


# for question, anwer in initial_faqs.items():
#     insert_faq(question, anwer)


# Preprocess the FAQs
# faq_keys = list(faqs.keys())
# faq_values = list(faqs.values())

# vectorize = TfidfVectorizer()
# faq_vector = vectorize.fit_transform(faq_values)

def get_answer(user_question, conversation_history=None):
  if conversation_history is None:
    conversation_history = []

  input_text = "".join(conversation_history)

  answer = text2text_generation_pipeline(
      f"Answer the following question: {user_question} give this conversation history {input_text}",
       max_new_tokens = 256,
       do_sample= True,
       temperature = 0.8
       )

  conversation_history.append(answer[0]['generated_text'])
  return answer[0]['generated_text'], conversation_history[0]


#   user_question = user_question.lower()
#   user_vector = vectorize.transform([user_question])
#   similarities = cosine_similarity(user_vector, faq_vector)
#   most_similar_index = np.argmax(similarities[0])
#   return faq_keys[most_similar_index]


@app.route("/", methods=["GET", "POST"])
def index():
    conversation_history = []

    if request.method == "POST":
        user_question = request.form["user_input"]
        answer, conversation_history = get_answer(user_question, conversation_history)
        audio_base64 = get_audio_base64(answer)
        print("-" * 20)
        print(conversation_history)
        print("-" * 20)
        return render_template(
            'index.html',
            answer=answer,
            question=user_question,
            audio_base64=audio_base64,
            conversation_history=conversation_history
            )
    return render_template('index.html', answer="", question="", audio_base64="", conversation_history=[])


if __name__ == "__main__":
    app.run(debug=True)  # debug=True for automatic reloading during development