from transformers import AutoTokenizer, T5ForConditionalGeneration

question = "Wie viel kostet eine Infefktionsschutzbelehrung?"

tokenizer = AutoTokenizer.from_pretrained("dwolpers/german_T5_Large_Closed_New")
model = T5ForConditionalGeneration.from_pretrained("dwolpers/german_T5_Large_Closed_New")
input_ids = tokenizer("Beantworte die Frage: " + question, return_tensors="pt").input_ids
gen_tokens = model.generate(input_ids, max_length=300)
print(tokenizer.batch_decode(gen_tokens, skip_special_tokens=True)[0])




question2 = "Wie viel kostet eine Infefktionsschutzbelehrung?"
context = "Infektionsschutzbelehrung inklusive Bescheinigung beantragen Kosten:\nBei Belehrung in Präsenz: Erkundigen Sie sich bei der Terminvereinbarung über die entstehenden Kosten. Diese können je nach Landkreis unterschiedlich ausfallen. Außerdem können je nach Zweck der Belehrung unterschiedlich hohe Gebühren anfallen. Rechnen Sie mit etwa 35 Euro. In einigen Landkreisen sind die Gebühren für Schüler Studenten und Auszubildende etwas geringer. Bei Online-Belehrung: Es können je nach zuständigem Gesundheitsamt und Zweck der Belehrung unterschiedlich hohe Gebühren anfallen. Diese werden Ihnen im Online-Antrag angezeigt und durch Online-Bezahlung beglichen.\n"

tokenizer2 = AutoTokenizer.from_pretrained("dwolpers/german_T5_Large_Open_New")
model2 = T5ForConditionalGeneration.from_pretrained("dwolpers/german_T5_Large_Open_New")
input_ids = tokenizer2("Beantworte mit folgendem Kontext die Frage: " + question2 + ' ' + context, return_tensors="pt").input_ids
gen_tokens = model2.generate(input_ids, max_length=300)
print(tokenizer2.batch_decode(gen_tokens, skip_special_tokens=True)[0])