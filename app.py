import os
import re
import json
from io import BytesIO
from datetime import datetime

from dotenv import load_dotenv

import streamlit as st
import streamlit.components.v1 as components
from openai import OpenAI

from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.utils import ImageReader
from PIL import Image

load_dotenv()

API_KEY = os.getenv("OPENROUTER_API_KEY")

if not API_KEY:
    st.error("OpenRouter API Key not found!\nPlease check your .env file.")
    st.stop()

client = OpenAI(
    api_key=API_KEY,
    base_url="https://openrouter.ai/api/v1",
    timeout=60
)

st.set_page_config(
    page_title="🎬 Generative AI-Based Filmmaking Assistant",
    page_icon="🎬",
    layout="wide"
)

st.title("🎬 Generative AI-Based Filmmaking Assistant")

st.markdown(
"""
Create simple, engaging movie stories with an AI director.

The assistant will ask only a few questions before writing your complete story.
"""
)

# Sidebar

with st.sidebar:

    st.header("🎥 Story Assistant")

    st.write(
        """
This assistant helps you create movie stories.

It will:

✅ Understand your idea

✅ Ask only a few short questions

✅ Write a complete story

✅ Let you download it as a PDF
"""
    )

    if st.button("🗑️ Start New Story"):

        st.session_state.messages = [
            {
                "role": "system",
                "content": ""
            }
        ]

        st.rerun()

SYSTEM_PROMPT = """
You are CineDirector AI.

You are an experienced, friendly and creative film director.

Your only job is helping users create movie stories.

If the user asks anything unrelated to creating a movie story, reply only:

I am sorry, I can only help you create a movie story. Please tell me about the story you want.

--------------------------------------------------

Your workflow

Step 1

Understand what the user already told you.

Possible information includes:

• Genre

• Main character

• Story idea

• Setting

• Ending preference

Never ask again for information already given.

--------------------------------------------------

Step 2

Ask only ONE short question at a time.

Never ask multiple questions together.

Keep every reply under two short sentences.

--------------------------------------------------

Step 3

Do NOT ask unnecessary questions.

You MUST stop asking questions once you know:

• Genre

• One main character

• One setting

• One main conflict or goal

After this, immediately write the story.

Never continue asking questions after enough information is available.

Maximum number of questions allowed:

FOUR

If you have already asked four questions,

generate the story immediately.

--------------------------------------------------

Important Rules

Never rename characters.

Never change the setting.

Never invent important information.

If something essential is missing,

ask for only that one thing.

If the user already provided many details,

skip questions and start writing.

Use only simple English.

Use short sentences.

Avoid difficult words.

--------------------------------------------------

When ready say exactly:

Great, I have what I need. Here's your story.

Then immediately write the story.

Do not add any introduction afterwards.

--------------------------------------------------

Story Format

Title: Story Name

Paragraph 1

Paragraph 2

Paragraph 3

Paragraph 4

Final lesson paragraph.

--------------------------------------------------

Do NOT generate

Scene headings

Camera directions

Screenplay format

INT./EXT.

Character profiles

Dialogue scripts

Poster ideas

Editing advice

Production planning

Cinematography explanations

Any language other than English

Anything unrelated to the story.

--------------------------------------------------

Example

Title: Belle and the Beast

Belle was a kind and brave girl. One day, her father became trapped inside a large castle owned by a frightening Beast. To save her father, Belle chose to stay in the castle instead.

At first, Belle feared the Beast. As time passed, she realised he was lonely rather than cruel. They slowly became close friends.

Later, Belle returned home to care for her sick father. The Beast allowed her to leave because he cared about her happiness.

When Belle finally returned, she found the Beast weak and close to death. She told him she loved him. The magic spell was broken, and the Beast became a handsome prince.

The story teaches us that kindness and love are more valuable than beauty or appearance.
"""

# Initialize Session State
# 

if "messages" not in st.session_state:

    st.session_state.messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT
        }
    ]

# Limit Chat History

MAX_HISTORY = 20

if len(st.session_state.messages) > MAX_HISTORY:

    st.session_state.messages = (
        [st.session_state.messages[0]]
        + st.session_state.messages[-19:]
    )

# Helper Function: Check if Response is a Complete Story

def is_story(content: str) -> bool:
    """
    Determine whether the assistant response is likely
    to be a completed story.
    """

    if not content:
        return False

    content_lower = content.lower()

    story_indicators = [
        "here's your story",
        "here is your story",
        "title:"
    ]

    if any(indicator in content_lower for indicator in story_indicators):
        return True

    # Long responses are usually complete stories
    if len(content.split()) >= 120:
        return True

    return False

# Helper Function: Extract Story Title

def extract_title(content: str) -> str:
    """
    Extract the story title from the AI response.
    """

    if not content:
        return "My Story"

    # Look for "Title: ..."
    match = re.search(
        r"^\s*Title\s*[:\-]\s*(.+)$",
        content,
        flags=re.IGNORECASE | re.MULTILINE
    )

    if match:
        title = match.group(1).strip()

        if title:
            return title

    # Otherwise use the first short non-empty line
    lines = [
        line.strip()
        for line in content.splitlines()
        if line.strip()
    ]

    for line in lines:

        if line.lower().startswith("great, i have"):
            continue

        if len(line.split()) <= 10:
            return line.strip("# ").strip()

    return "My Story"


# Helper Function: Safe Filename

def create_safe_filename(title: str) -> str:
    """
    Convert title into a safe filename.
    """

    filename = re.sub(
        r"[^A-Za-z0-9 _-]",
        "",
        title
    )

    filename = filename.strip()

    filename = filename.replace(" ", "_")

    filename = filename[:80]

    if not filename:
        filename = "story"

    return filename + ".pdf"

# Helper Function:
# Get Title + Filename

def extract_title_and_filename(content: str):

    title = extract_title(content)

    filename = create_safe_filename(title)

    return title, filename


# Helper Function: Clean Story Body
# Removes the "Title:" line and the intro line so they are not duplicated inside the PDF body, since the title is already rendered in the PDF header

def clean_story_body(content: str) -> str:

    lines = content.splitlines()
    cleaned_lines = []

    for line in lines:

        stripped = line.strip()
        lower = stripped.lower()

        if lower.startswith("title"):
            continue

        if lower.startswith("great, i have"):
            continue

        cleaned_lines.append(line)

    return "\n".join(cleaned_lines).strip()


# Helper Function: Build Watermark Image (with reduced opacity)

def get_watermark_image_buffer(logo_path: str, opacity: float = 0.12):
    """
    Opens the logo image, reduces its opacity, and returns
    an in-memory PNG (with alpha channel) suitable for use
    as a translucent watermark in the PDF.
    """

    if not os.path.exists(logo_path):
        return None

    try:
        img = Image.open(logo_path).convert("RGBA")

        alpha = img.split()[3].point(lambda p: int(p * opacity))
        img.putalpha(alpha)

        buffer = BytesIO()
        img.save(buffer, format="PNG")
        buffer.seek(0)

        return buffer

    except Exception:
        return None

# Helper Function: Generate PDF for the Story

LOGO_PATH = "logo.jpeg"

def generate_story_pdf(title: str, content: str, logo_path: str = LOGO_PATH) -> bytes:
    """
    Builds a PDF containing:
      - The story title and the current date/time at the top
      - A faded watermark image (logo.jpeg) on every page
      - The story text as the body
    Returns the PDF as raw bytes, ready for st.download_button.
    """

    buffer = BytesIO()

    now_str = datetime.now().strftime("%d %B %Y, %I:%M %p")

    watermark_buffer = get_watermark_image_buffer(logo_path)

    def draw_watermark_and_header(canvas, doc):

        canvas.saveState()

        page_w, page_h = A4

        # Watermark (centered, faded)
        if watermark_buffer is not None:
            try:
                watermark_buffer.seek(0)
                img_reader = ImageReader(watermark_buffer)
                iw, ih = img_reader.getSize()

                target_w = page_w * 0.6
                scale = target_w / iw
                w = iw * scale
                h = ih * scale

                x = (page_w - w) / 2
                y = (page_h - h) / 2

                canvas.drawImage(
                    img_reader,
                    x, y,
                    width=w, height=h,
                    mask="auto",
                    preserveAspectRatio=True
                )
            except Exception:
                pass

        # Header: Title + Date/Time
        canvas.setFont("Helvetica-Bold", 18)
        canvas.drawCentredString(page_w / 2, page_h - 55, title)

        canvas.setFont("Helvetica", 10)
        canvas.drawCentredString(page_w / 2, page_h - 72, now_str)

        canvas.setStrokeColorRGB(0.6, 0.6, 0.6)
        canvas.line(50, page_h - 82, page_w - 50, page_h - 82)

        canvas.restoreState()

    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        topMargin=100,
        bottomMargin=50,
        leftMargin=55,
        rightMargin=55
    )

    styles = getSampleStyleSheet()

    body_style = ParagraphStyle(
        "StoryBody",
        parent=styles["Normal"],
        fontSize=12,
        leading=18,
        spaceAfter=12,
        alignment=4  # justified
    )

    flowables = []

    cleaned_content = clean_story_body(content)

    for para in cleaned_content.split("\n\n"):

        para = para.strip()

        if not para:
            continue

        safe_para = (
            para
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace("\n", "<br/>")
        )

        flowables.append(Paragraph(safe_para, body_style))
        flowables.append(Spacer(1, 6))

    if not flowables:
        flowables.append(Paragraph("(No story content)", body_style))

    doc.build(
        flowables,
        onFirstPage=draw_watermark_and_header,
        onLaterPages=draw_watermark_and_header
    )

    buffer.seek(0)
    return buffer.read()


# Helper Function: Voice Playback Button
# Uses the browser's built-in Speech Synthesis API, So it works instantly with no extra API calls or cost

def render_voice_button(text: str, key_suffix: str):

    safe_text_json = json.dumps(text)
    play_id = f"cinedirector_play_{key_suffix}"
    stop_id = f"cinedirector_stop_{key_suffix}"

    html_code = f"""
    <div style="display:flex; gap:8px; margin-top:6px;">
        <button id="{play_id}" style="
            padding:6px 14px;
            border-radius:6px;
            border:none;
            background:#4CAF50;
            color:white;
            cursor:pointer;
            font-size:14px;">
            🔊 Listen
        </button>
        <button id="{stop_id}" style="
            padding:6px 14px;
            border-radius:6px;
            border:none;
            background:#B33A3A;
            color:white;
            cursor:pointer;
            font-size:14px;">
            ⏹ Stop
        </button>
    </div>
    <script>
        (function() {{
            const text = {safe_text_json};
            const playBtn = document.getElementById("{play_id}");
            const stopBtn = document.getElementById("{stop_id}");

            playBtn.addEventListener("click", function() {{
                window.speechSynthesis.cancel();
                const utterance = new SpeechSynthesisUtterance(text);
                utterance.rate = 1;
                utterance.pitch = 1;
                utterance.lang = "en-US";
                window.speechSynthesis.speak(utterance);
            }});

            stopBtn.addEventListener("click", function() {{
                window.speechSynthesis.cancel();
            }});
        }})();
    </script>
    """

    components.html(html_code, height=50)

# Render Assistant Message

def render_assistant_message(
    content: str,
    key_suffix: str
):

    st.markdown(content)

    # Voice playback button for every assistant reply
    render_voice_button(content, key_suffix)

    # Download button only for completed stories (as PDF)
    if is_story(content):

        title, filename = extract_title_and_filename(content)

        pdf_bytes = generate_story_pdf(title, content)

        st.download_button(
            label=f'⬇️ Download "{title}" (PDF)',
            data=pdf_bytes,
            file_name=filename,
            mime="application/pdf",
            key=f"download_{key_suffix}"
        )

# Display Chat History

for index, message in enumerate(
    st.session_state.messages
):

    if message["role"] == "system":
        continue

    with st.chat_message(message["role"]):

        if message["role"] == "assistant":

            render_assistant_message(
                message["content"],
                str(index)
            )

        else:

            st.markdown(
                message["content"]
            )

# Chat Input

prompt = st.chat_input(
    "Tell me about the movie story you'd like to create..."
)

# Process User Input

if prompt:

    # Save User Message

    st.session_state.messages.append(
        {
            "role": "user",
            "content": prompt
        }
    )

    with st.chat_message("user"):
        st.markdown(prompt)

    # Assistant Response

    with st.chat_message("assistant"):

        with st.spinner("🎬 Director is writing..."):

            try:

                response = client.chat.completions.create(

                    model="meta-llama/llama-3.3-70b-instruct",

                    messages=st.session_state.messages,

                    temperature=0.7,

                    max_tokens=1000,

                    extra_headers={
                        "HTTP-Referer": "http://localhost:8501",
                        "X-Title": "Generative AI-Based Filmmaking Assistant"
                    }

                )

                answer = response.choices[0].message.content

                if answer is None:
                    answer = (
                        "Sorry, I couldn't generate a response. "
                        "Please try again."
                    )

                answer = answer.strip()

            except Exception as e:

                answer = (
                    "❌ Sorry, something went wrong while "
                    "contacting the AI service.\n\n"
                    f"Error Details:\n{e}"
                )

        # Display Assistant Response

        render_assistant_message(
            answer,
            key_suffix=f"new_{len(st.session_state.messages)}"
        )

    # Save Assistant Response

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer
        }
    )

# Footer

st.markdown("---")

st.caption(
    "🎬 Generative AI-Based Filmmaking Assistant | "
    "Powered by Wikpolt Softwares Pvt Ltd."
)