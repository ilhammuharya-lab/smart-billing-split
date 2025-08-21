# bill_splitter.py
import streamlit as st
import torch
from transformers import AutoProcessor, AutoModelForVision2Seq
from PIL import Image
import xmltodict
import re
import json
from io import BytesIO
import time
import subprocess
import sys

# Set page config
st.set_page_config(
    page_title="Bill Splitter",
    page_icon="🧾",
    layout="wide"
)

# Initialize session state
if 'parsed_data' not in st.session_state:
    st.session_state.parsed_data = None
if 'people' not in st.session_state:
    st.session_state.people = []
if 'assignments' not in st.session_state:
    st.session_state.assignments = {}
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False
if 'processor' not in st.session_state:
    st.session_state.processor = None
if 'model' not in st.session_state:
    st.session_state.model = None

def install_sentencepiece():
    """Install sentencepiece if not available"""
    try:
        import sentencepiece
        return True
    except ImportError:
        st.warning("Installing sentencepiece...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "sentencepiece"])
            import sentencepiece
            return True
        except:
            st.error("Failed to install sentencepiece. Please run: pip install sentencepiece")
            return False

@st.cache_resource
def load_model():
    """Load the Donut model and processor"""
    # First ensure sentencepiece is available
    if not install_sentencepiece():
        return None, None
    
    model_name = "naver-clova-ix/donut-base-finetuned-cord-v2"
    try:
        # Try explicit DonutProcessor first
        from transformers import DonutProcessor, VisionEncoderDecoderModel
        processor = DonutProcessor.from_pretrained(model_name)
        model = VisionEncoderDecoderModel.from_pretrained(model_name)
        st.session_state.model_loaded = True
        return processor, model
    except Exception as e:
        st.error(f"Error loading model with DonutProcessor: {e}")
        # Fallback to Auto classes
        try:
            processor = AutoProcessor.from_pretrained(model_name)
            model = AutoModelForVision2Seq.from_pretrained(model_name)
            st.session_state.model_loaded = True
            return processor, model
        except Exception as e2:
            st.error(f"Fallback loading also failed: {e2}")
            return None, None

def extract_from_raw_output(raw_output):
    """Manual extraction dari raw model output"""
    result = {'s_cord-v2': {'s_menu': {'s_nm': [], 's_cnt': [], 's_price': []}}}
    
    try:
        # Extract items dengan pattern matching
        lines = raw_output.split('<sep/>')
        for line in lines:
            # Cari pola: <s_nm>...</s_nm>
            nm_match = re.search(r'<s_nm>(.*?)</s_nm>', line)
            if nm_match:
                item_name = nm_match.group(1).strip()
                
                # Cari quantity
                cnt_match = re.search(r'<s_cnt>(.*?)</s_cnt>', line)
                quantity = cnt_match.group(1).strip() if cnt_match else '1'
                
                # Cari price
                price_match = re.search(r'<s_price>(.*?)</s_price>', line)
                price = price_match.group(1).strip() if price_match else '0'
                
                # Hanya tambahkan jika nama item tidak kosong dan bukan metadata
                if item_name and len(item_name) > 3 and not any(x in item_name for x in ['Rcpt#', 'POS', 'TABLE', 'Pax']):
                    result['s_cord-v2']['s_menu']['s_nm'].append(item_name)
                    result['s_cord-v2']['s_menu']['s_cnt'].append(quantity)
                    result['s_cord-v2']['s_menu']['s_price'].append(price)
        
        # Extract totals dari seluruh text
        total_match = re.search(r'<s_total_price>(.*?)</s_total_price>', raw_output)
        subtotal_match = re.search(r'<s_subtotal_price>(.*?)</s_subtotal_price>', raw_output)
        tax_match = re.search(r'<s_tax_price>(.*?)</s_tax_price>', raw_output)
        service_match = re.search(r'<s_service_price>(.*?)</s_service_price>', raw_output)
        
        # Juga cari pola angka besar yang mungkin adalah totals
        large_numbers = re.findall(r'[0-9,]{6,}', raw_output)  # angka dengan 6+ digit/karakter
        if large_numbers:
            large_numbers = [num for num in large_numbers if ',' in num and len(num) > 5]
            if large_numbers:
                largest = max(large_numbers, key=lambda x: int(x.replace(',', '')))
                if not total_match:
                    result['s_cord-v2']['s_total'] = {'s_total_price': largest}
        
        if total_match:
            result['s_cord-v2']['s_total'] = {'s_total_price': total_match.group(1).strip()}
        if subtotal_match:
            if 's_sub_total' not in result['s_cord-v2']:
                result['s_cord-v2']['s_sub_total'] = {}
            result['s_cord-v2']['s_sub_total']['s_subtotal_price'] = subtotal_match.group(1).strip()
        if tax_match:
            if 's_sub_total' not in result['s_cord-v2']:
                result['s_cord-v2']['s_sub_total'] = {}
            result['s_cord-v2']['s_sub_total']['s_tax_price'] = tax_match.group(1).strip()
        if service_match:
            if 's_sub_total' not in result['s_cord-v2']:
                result['s_cord-v2']['s_sub_total'] = {}
            result['s_cord-v2']['s_sub_total']['s_service_price'] = service_match.group(1).strip()
            
    except Exception as e:
        st.error(f"Manual extraction failed: {e}")
    
    return result

def parse_receipt(image, processor, model):
    """Parse receipt image menggunakan manual extraction dari raw output"""
    try:
        # Preprocess image
        decoder_input_ids = processor.tokenizer(
            "<s_cord-v2>", add_special_tokens=False
        ).input_ids
        decoder_input_ids = torch.tensor(decoder_input_ids).unsqueeze(0)
        pixel_values = processor(image, return_tensors="pt").pixel_values

        # Generate output
        generation_output = model.generate(
            pixel_values,
            decoder_input_ids=decoder_input_ids,
            max_length=model.decoder.config.max_position_embeddings,
            pad_token_id=processor.tokenizer.pad_token_id,
            eos_token_id=processor.tokenizer.eos_token_id,
            use_cache=True,
            num_beams=3,
            early_stopping=True,
            bad_words_ids=[[processor.tokenizer.unk_token_id]],
            return_dict_in_generate=True,
        )

        # Decode sequence
        decoded_sequence = processor.batch_decode(generation_output.sequences)[0]
        decoded_sequence = decoded_sequence.replace(processor.tokenizer.eos_token, "")
        decoded_sequence = decoded_sequence.replace(processor.tokenizer.pad_token, "")
        
        # Tampilkan raw output untuk debugging
        st.write("🔍 Raw Model Output:", decoded_sequence)
        
        # Langsung gunakan manual extraction tanpa XML parsing
        parsed_data = extract_from_raw_output(decoded_sequence)
        return parsed_data
        
    except Exception as e:
        st.error(f"Error parsing receipt: {e}")
        # Return empty structure instead of None
        return {'s_cord-v2': {'s_menu': {'s_nm': [], 's_cnt': [], 's_price': []}}}

def create_item(name, quantity, price):
    """Create a standardized item dictionary"""
    return {
        'name': str(name) if name else 'Unknown Item',
        'quantity': str(quantity) if quantity else '1',
        'price': str(price) if price else '0'
    }

def extract_items(parsed_data):
    """Extract items from parsed data dengan pengecekan null yang lebih ketat"""
    items = []
    
    try:
        # Check if parsed_data is None or empty
        if not parsed_data:
            st.warning("No parsed data available")
            return items
            
        # Check if the expected structure exists
        if 's_cord-v2' not in parsed_data:
            st.warning("Expected structure 's_cord-v2' not found in parsed data")
            return items
            
        cord_data = parsed_data['s_cord-v2']
        
        if not cord_data:
            return items
        
        # Extract from menu structure
        if 's_menu' in cord_data:
            menu_data = cord_data['s_menu']
            if menu_data and 's_nm' in menu_data:
                names = menu_data['s_nm']
                quantities = menu_data.get('s_cnt', [])
                prices = menu_data.get('s_price', [])
                
                if isinstance(names, list):
                    for i, name in enumerate(names):
                        quantity = quantities[i] if i < len(quantities) else '1'
                        price = prices[i] if i < len(prices) else '0'
                        items.append(create_item(name, quantity, price))
                else:
                    # Single item
                    quantity = quantities if isinstance(quantities, str) else '1'
                    price = prices if isinstance(prices, str) else '0'
                    items.append(create_item(names, quantity, price))
        
        return items
        
    except Exception as e:
        st.error(f"Error extracting items: {e}")
        return []

def extract_totals(parsed_data):
    """Extract totals from parsed data dengan pengecekan null"""
    totals = {}
    
    try:
        if not parsed_data or 's_cord-v2' not in parsed_data:
            return totals
            
        cord_data = parsed_data['s_cord-v2']
        
        if not cord_data:
            return totals
        
        # Extract from sub_total
        if 's_sub_total' in cord_data:
            sub_total = cord_data['s_sub_total']
            if sub_total:
                totals['subtotal'] = sub_total.get('s_subtotal_price', '0')
                totals['service'] = sub_total.get('s_service_price', '0')
                totals['tax'] = sub_total.get('s_tax_price', '0')
        
        # Extract from total
        if 's_total' in cord_data:
            total_data = cord_data['s_total']
            if total_data:
                totals['total'] = total_data.get('s_total_price', '0')
                
    except Exception as e:
        st.error(f"Error extracting totals: {e}")
    
    # Set default values if missing
    if 'subtotal' not in totals:
        totals['subtotal'] = '0'
    if 'service' not in totals:
        totals['service'] = '0'
    if 'tax' not in totals:
        totals['tax'] = '0'
    if 'total' not in totals:
        totals['total'] = '0'
    
    return totals

def convert_price(price_str):
    """Convert price string to float"""
    try:
        # Remove commas and any non-numeric characters except decimal point
        clean_str = re.sub(r'[^\d.]', '', price_str)
        return float(clean_str)
    except:
        return 0.0

def calculate_individual_totals(people, items, assignments, totals):
    """Calculate how much each person should pay"""
    person_totals = {person: 0.0 for person in people}
    
    # Calculate item costs
    for i, item in enumerate(items):
        item_price = convert_price(item['price'])
        item_quantity = convert_price(item['quantity'])
        total_item_cost = item_price * item_quantity
        
        # Find who is assigned to this item
        assigned_person = assignments.get(str(i), None)
        if assigned_person and assigned_person in people:
            person_totals[assigned_person] += total_item_cost
    
    # Calculate additional charges per person (proportional to their share of subtotal)
    subtotal = convert_price(totals.get('subtotal', '0'))
    service = convert_price(totals.get('service', '0'))
    tax = convert_price(totals.get('tax', '0'))
    total = convert_price(totals.get('total', '0'))
    
    if subtotal > 0:
        for person in people:
            if person_totals[person] > 0:
                # Calculate proportion of subtotal
                proportion = person_totals[person] / subtotal
                person_totals[person] += service * proportion + tax * proportion
    
    return person_totals

def main():
    st.title("🧾 Bill Splitter")
    st.write("Upload a receipt image, and we'll help you split the bill among your friends!")
    
    # Load model (cached)
    if st.session_state.processor is None or st.session_state.model is None:
        with st.spinner("Loading AI model... This may take a few minutes."):
            st.session_state.processor, st.session_state.model = load_model()
    
    processor = st.session_state.processor
    model = st.session_state.model
    
    # Show warning if model not loaded
    if processor is None or model is None:
        st.error("Model failed to load. Please check the requirements and try again.")
        st.info("Make sure you have installed all dependencies. Check the console for errors.")
        return
    
    # File upload
    uploaded_file = st.file_uploader("Choose a receipt image", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Display image
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Receipt", use_column_width=True)
        
        # Parse button
        if st.button("Parse Receipt"):
            with st.spinner("Parsing receipt... This may take a while on CPU."):
                start_time = time.time()
                parsed_data = parse_receipt(image, processor, model)
                processing_time = time.time() - start_time
                
                if parsed_data:
                    st.session_state.parsed_data = parsed_data
                    st.success(f"Receipt parsed successfully! (Time: {processing_time:.2f}s)")
                    
                    # Debug info untuk melihat struktur data
                    st.write("📊 Parsed data structure:", type(parsed_data))
                    
                    # Display raw parsed data (for debugging)
                    with st.expander("View parsed data"):
                        st.json(parsed_data)
    
    # If we have parsed data, show the items and allow assignment
    if st.session_state.parsed_data:
        parsed_data = st.session_state.parsed_data
        
        # Extract items and totals
        items = extract_items(parsed_data)
        totals = extract_totals(parsed_data)
        
        # People input
        st.subheader("👥 People")
        people_input = st.text_input("Enter names separated by commas (e.g., Alice, Bob, Charlie)")
        if people_input:
            st.session_state.people = [name.strip() for name in people_input.split(",") if name.strip()]
        
        if st.session_state.people:
            st.write("Participants:", ", ".join(st.session_state.people))
            
            if items:
                # Items and assignment
                st.subheader("🛒 Items")
                for i, item in enumerate(items):
                    col1, col2, col3, col4 = st.columns([3, 1, 1, 2])
                    with col1:
                        st.text_input("Item", value=item['name'], key=f"name_{i}", disabled=True)
                    with col2:
                        st.text_input("Qty", value=item['quantity'], key=f"qty_{i}", disabled=True)
                    with col3:
                        st.text_input("Price", value=item['price'], key=f"price_{i}", disabled=True)
                    with col4:
                        # Calculate total for this item
                        item_total = convert_price(item['price']) * convert_price(item['quantity'])
                        st.session_state.assignments[str(i)] = st.selectbox(
                            "Who paid?",
                            options=[""] + st.session_state.people,
                            key=f"assign_{i}"
                        )
                        st.write(f"Total: {item_total:,.0f}")
                
                # Display totals
                st.subheader("💰 Totals")
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Subtotal", totals.get('subtotal', '0'))
                with col2:
                    st.metric("Service", totals.get('service', '0'))
                with col3:
                    st.metric("Tax", totals.get('tax', '0'))
                with col4:
                    st.metric("Total", totals.get('total', '0'), delta_color="off")
                
                # Calculate and display individual shares
                if st.button("Calculate Shares"):
                    person_totals = calculate_individual_totals(
                        st.session_state.people, items, st.session_state.assignments, totals
                    )
                    
                    st.subheader("🧾 Individual Shares")
                    total_calculated = sum(person_totals.values())
                    expected_total = convert_price(totals.get('total', '0'))
                    
                    for person, amount in person_totals.items():
                        st.metric(person, f"{amount:,.0f}")
                    
                    # Check if our calculation matches the receipt total
                    if abs(total_calculated - expected_total) > 1:
                        st.warning(f"Note: Calculated total ({total_calculated:,.0f}) doesn't match receipt total ({expected_total:,.0f}). There might be missing items or incorrect parsing.")
            else:
                st.warning("No items found in the receipt. Please try with a different image.")

if __name__ == "__main__":
    main()