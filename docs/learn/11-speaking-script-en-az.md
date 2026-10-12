# My project: what I will say (simple English / sadə Azərbaycan dili)

## 1. One-minute version
**EN:** My project is an AI assistant for AML and KYC. It answers questions from compliance documents and shows the source. It reads invoices and checks the data. It finds the company in a list. An agent decides which tool to use. I measured the results, and I know the weak points.

**AZ:** Mənim layihəm AML və KYC üzrə süni intellekt köməkçisidir. O, sənədlərə əsasən suallara cavab verir və mənbəni göstərir. Hesab-fakturaları oxuyur və məlumatları yoxlayır. Şirkəti siyahıda tapır. Agent hansı aləti istifadə edəcəyini özü seçir. Nəticələri ölçmüşəm və zəif tərəfləri bilirəm.

## 2. Questions and answers (RAG)
**EN:** First, I cut the documents into small pieces. I turn each piece into numbers. This is called an embedding. When a question comes, I find the 8 most similar pieces. The model answers only from these pieces and shows the source. If the answer is not there, it says "I don't know". So the model does not invent things.

**AZ:** Əvvəlcə sənədləri kiçik hissələrə bölürəm. Hər hissəni rəqəmlərə çevirirəm, buna embedding deyilir. Sual gələndə ən oxşar 8 hissəni tapıram. Model yalnız bu hissələrə əsasən cavab verir və mənbəni göstərir. Cavab orada yoxdursa, "bilmirəm" deyir. Beləliklə model uydurmur.

## 3. Invoices
**EN:** If the PDF has text, I take the text. If it is a scan, I read it with OCR. I use Tesseract for that. Then the model returns the data as JSON: company, VÖEN, number, date and amount. I check it with rules: VÖEN has 10 digits, the amount is more than zero, the date is not in the future. If there is a mistake, the model tries to fix it two times. If it still fails, a person checks the invoice.

**AZ:** PDF-də mətn varsa, mətni birbaşa götürürəm. Skan şəkildirsə, OCR ilə oxuyuram, bunun üçün Tesseract istifadə edirəm. Sonra model məlumatları JSON formatında qaytarır: şirkət, VÖEN, nömrə, tarix və məbləğ. Mən qaydalarla yoxlayıram: VÖEN 10 rəqəmdir, məbləğ sıfırdan böyükdür, tarix gələcəkdə deyil. Səhv varsa, model iki dəfə düzəltməyə çalışır. Yenə alınmırsa, hesab-fakturanı insan yoxlayır.

## 4. Finding the company
**EN:** I search step by step. First by VÖEN. Then by the exact name. Then by a similar name, to catch typos. Then by similar meaning. If I am not sure, a person decides. In finance, choosing the wrong company is worse than not finding it.

**AZ:** Addım-addım axtarıram. Əvvəlcə VÖEN ilə. Sonra dəqiq adla. Sonra oxşar adla, yazı səhvlərini tutmaq üçün. Sonra məna oxşarlığı ilə. Əmin deyiləmsə, insan qərar verir. Maliyyədə səhv şirkəti seçmək onu tapmamaqdan daha təhlükəlidir.

## 5. The agent
**EN:** The agent reads the question and chooses a tool: the knowledge base, a calculator, web search, SQL or invoice extraction. SQL is read-only. It cannot delete or change anything.

**AZ:** Agent sualı oxuyur və aləti özü seçir: bilik bazası, kalkulyator, internetdə axtarış, SQL və ya hesab-fakturadan məlumat çıxarmaq. SQL yalnız oxumaq üçündür. Heç nəyi silə və ya dəyişə bilməz.

## 6. Reliable model calls
**EN:** All model calls go through one place. A short, simple request goes to a small model, because it is cheaper and faster. Harder work goes to the big model. If there is an error, it waits a little and tries again. If the provider does not work, it switches to another provider. If nothing works, the request goes to a person. I log the time, the tokens and the cost of every call.

**AZ:** Bütün model çağırışları bir yerdən keçir. Qısa və sadə sorğu kiçik modelə gedir, çünki daha ucuz və sürətlidir. Çətin iş böyük modelə gedir. Xəta olanda bir az gözləyir və yenidən cəhd edir. Provayder işləmirsə, başqa provayderə keçir. Heç biri işləmirsə, sorğu insana gedir. Hər çağırışın vaxtını, tokenlərini və qiymətini yazıram.

## 7. Other parts
**EN:** Data collection: Celery and Redis collect web pages automatically at set times. Duplicates are removed, and new pieces go into the knowledge base. Voice: Whisper turns speech into text, and then the agent answers. Streaming: the answer appears part by part, like in a chat. Translation and summary: the model translates or shortens a text, and a glossary keeps the terms correct. MCP: other AI apps can use my tools in a standard way.

**AZ:** Məlumat toplama: Celery və Redis saytlardan məlumatı müəyyən vaxtlarda avtomatik toplayır. Təkrarlananlar atılır, yeni hissələr bilik bazasına əlavə olunur. Səs: Whisper səsi mətnə çevirir, sonra agent cavab verir. Streaming: cavab çatda olduğu kimi hissə-hissə görünür. Tərcümə və xülasə: model mətni tərcümə edir və ya qısaldır, lüğət isə terminləri düzgün saxlayır. MCP: başqa süni intellekt tətbiqləri mənim alətlərimdən standart yolla istifadə edə bilir.

## 8. Programs I used and why
**EN:** Python is the main language. FastAPI is the server. Streamlit is the demo screen. LangChain runs the agent. Chroma or PostgreSQL with pgvector stores the vectors. Pydantic checks the invoice data. Tesseract reads scans. Celery and Redis run background jobs. Docker packs everything, and GitHub Actions runs 249 tests on every change.

**AZ:** Python əsas dildir. FastAPI serverdir. Streamlit demo ekranıdır. LangChain agenti işlədir. Vektorları Chroma və ya pgvector ilə PostgreSQL saxlayır. Pydantic hesab-faktura məlumatlarını yoxlayır. Tesseract skanları oxuyur. Celery və Redis arxa fonda işləri yerinə yetirir. Docker hər şeyi paketləyir, GitHub Actions isə hər dəyişiklikdə 249 testi işlədir.

## 9. How I choose a model
**EN:** I choose the model by the task. I start with the smallest model and test it. If the quality is not enough, I take a bigger one. I look at quality, speed and cost together. For answers I use gpt-oss-120b on Groq: it is fast and free to start. For short tasks I use the small gpt-oss-20b. The embedding model runs on my computer: it is free and the data does not leave. For speech I use Whisper base, because it works on a normal laptop.

**AZ:** Modeli tapşırığa görə seçirəm. Əvvəlcə ən kiçik modeli yoxlayıram. Keyfiyyət çatmırsa, daha böyüyünü götürürəm. Keyfiyyətə, sürətə və qiymətə birlikdə baxıram. Cavablar üçün Groq-da gpt-oss-120b istifadə edirəm: sürətlidir və başlamaq üçün pulsuzdur. Qısa tapşırıqlar üçün kiçik gpt-oss-20b modelini istifadə edirəm. Embedding modeli mənim kompüterimdə işləyir: pulsuzdur və məlumat kənara çıxmır. Səs üçün Whisper base istifadə edirəm, çünki adi noutbukda işləyir.

## 10. Results
**EN:** I measured everything I could. In the RAGAS test, "I don't know" answers went down from 5 to 1 out of 15 questions. On invoices, the main fields were 100% correct, but the invoices are synthetic. Company matching made no wrong matches on 52 test pairs. Speech recognition: English is good, Azerbaijani is weaker.

**AZ:** Ölçə bildiyim hər şeyi ölçmüşəm. RAGAS testində "bilmirəm" cavabları 15 sualdan 5-dən 1-ə düşdü. Hesab-fakturalarda əsas sahələr 100% düzgün oldu, amma hesab-fakturalar sintetikdir. Şirkət uyğunlaşdırma 52 test cütündə heç bir səhv uyğunluq vermədi. Səs tanıma: ingilis dili yaxşıdır, Azərbaycan dili zəifdir.

## 11. Weak points and next steps
**EN:** The invoice and company data is synthetic. I did not test it on real documents yet. Azerbaijani speech recognition is weaker. Next, I would test on 50 to 100 real anonymous invoices and use a multilingual embedding model.

**AZ:** Hesab-faktura və şirkət məlumatları sintetikdir. Hələ real sənədlərdə yoxlamamışam. Azərbaycan dilində səs tanıma zəifdir. Növbəti addım: 50-100 real, anonimləşdirilmiş hesab-faktura ilə yoxlamaq və çoxdilli embedding modeli istifadə etmək.
