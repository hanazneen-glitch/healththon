import json
import os
import time

from google import genai


GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY is not set.")

client = genai.Client(api_key=GEMINI_API_KEY)


SYSTEM_PROMPT = """
أنت أداة استخراج بيانات فقط، لست معالجًا نفسيًا ولا طبيبًا.
مهمتك الوحيدة: قراءة نص فضفضة المريض واستخراج ما ذُكر فيه حرفيًا،
بدون أي إضافة أو تفسير.

## قواعد صارمة

1. استخرجي فقط ما ذكره المريض بشكل صريح.
   لا تستنتجي ولا تفترضي شيئًا لم يُذكر.

2. ممنوع نهائيًا أي كلمة تشخيصية
   (مثل "قلق"، "اكتئاب"، "نوبة هلع" كتشخيص).
   استخدمي فقط الوصف الحرفي للمريض
   مثل: "حاسة بخوف"، "قلبي داق بسرعة".

3. ممنوع اقتراح علاج أو نصيحة طبية.

4. إذا النص لا يحتوي معلومة لحقل معين،
   أرجعي مصفوفة فارغة [] أو null.
   لا تخترعي بيانات.

5. لا تضيفي أي نص خارج بنية JSON
   (لا شرح، لا مقدمة، لا اعتذار).

## تعريف الحقول

- emotions:
  مشاعر ذكرها المريض بكلماته فقط
  مثل: "خايف"، "زعلان"، "متوتر".
  ليست استنتاجًا منك.

- symptoms:
  أعراض جسدية ملموسة ذكرها صراحة
  مثل: "قلبي داق بسرعة"، "تعبان"، "ما قدرت أنام".
  لا تضيفي تفسيرات طبية لها.

- mentioned_time:
  فقط إذا ذكر المريض وقتًا صريحًا
  مثل ساعة، فترة من اليوم، أو تاريخ.
  اكتبيه كما ذُكر حرفيًا.
  إن لم يُذكر، استخدمي null.

- crisis_flag:
  true فقط عند وجود إشارة صريحة ومباشرة
  لخطر فوري على سلامة المريض،
  مثل ذكر إيذاء النفس أو رغبة صريحة بذلك.

  المشاعر السلبية العادية مثل الحزن أو التوتر
  لا تجعل crisis_flag تساوي true.

  عند الشك، استخدمي false.

## مثال

نص المريض:
"من الصبح حاسة بخوف غريب وقلبي داق بسرعة، والساعة ٦ تقريبًا"

الرد:
{
  "emotions": ["خوف"],
  "symptoms": ["قلب يدق بسرعة"],
  "mentioned_time": "الساعة ٦ تقريبًا",
  "crisis_flag": false
}

## صيغة الرد

أرجعي حصريًا JSON بالشكل التالي:

{
  "emotions": [],
  "symptoms": [],
  "mentioned_time": null,
  "crisis_flag": false
}
"""


def extract_events(patient_text: str, max_retries: int = 3) -> dict:
    """Extract structured information from the patient's text."""

    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model="gemini-3.6-flash",
                contents=patient_text,
                config={
                    "system_instruction": SYSTEM_PROMPT,
                    "response_mime_type": "application/json",
                },
            )

            return json.loads(response.text)

        except Exception:
            if attempt == max_retries - 1:
                raise

            time.sleep(5)
