import os
import pickle
import pandas as pd
import numpy as np
from transformers import (
    AutoTokenizer, 
    AutoModelForQuestionAnswering, 
    pipeline
)

class PowerConsumptionAssistant:
    def __init__(self, csv_path, model_path=None):
        # Load the CSV data
        try:
            self.df = pd.read_csv(csv_path)
            self.df['start_time'] = pd.to_datetime(self.df['start_time'])
        except Exception as e:
            raise ValueError(f"Error loading CSV file: {e}")
        
        # Preprocess the dataset
        self.preprocess_data()
        
        # Model and tokenizer setup
        self.model_name = "deepset/roberta-base-squad2"  # More standard QA model
        
        # Load tokenizer
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        except Exception as e:
            raise ValueError(f"Error loading tokenizer: {e}")
        
        # Load or initialize model
        try:
            if model_path and os.path.exists(model_path):
                with open(model_path, 'rb') as f:
                    self.model = pickle.load(f)
            else:
                self.model = AutoModelForQuestionAnswering.from_pretrained(self.model_name)
        except Exception as e:
            print(f"Error loading model: {e}")
            self.model = AutoModelForQuestionAnswering.from_pretrained(self.model_name)
        
        # Create question-answering pipeline
        self.qa_pipeline = pipeline(
            "question-answering", 
            model=self.model, 
            tokenizer=self.tokenizer
        )

    def get_power_consumption_data(self):
        """
        Prepare power consumption data for chart rendering
        """
        devices = list(self.total_power_consumption.index)
        consumption = list(self.total_power_consumption.values)
        return {
            'devices': devices,
            'consumption': consumption
        }
    
    def preprocess_data(self):
        """
        Preprocess the power consumption dataset
        - Calculate total power consumption for each device
        - Create a summary context
        """
        # Group by device and calculate total average power consumption
        try:
            # Group by device and sum the average power
            self.total_power_consumption = self.df.groupby('device')['avg_power'].sum()
            
            # Sort devices by total power consumption
            self.sorted_appliances = self.total_power_consumption.sort_values(ascending=False)

            # Calculate additional metrics
            self.peak_power = self.df.groupby('device')['max_power'].max()
            self.min_power = self.df.groupby('device')['min_power'].min()
            self.power_variation = self.peak_power - self.min_power
            
            # Room-based calculations
            self.room_power = self.df.groupby('room')['avg_power'].sum()
            self.devices_by_room = {
                room: list(devices) 
                for room, devices in self.df.groupby('room')['device'].unique().items()
            }
            
            # Calculate total power for percentage calculations
            self.total_power = self.df['avg_power'].sum()
            
            # Create enhanced context
            self.context = self._create_enhanced_context()
        
        except Exception as e:
            raise ValueError(f"Error in preprocessing: {e}")
        
    def _create_enhanced_context(self):
        """Create a comprehensive context with all metrics"""
        return {
            'total_power': self.total_power,
            'devices': dict(self.sorted_appliances),
            'peak_power': dict(self.peak_power),
            'min_power': dict(self.min_power),
            'power_variation': dict(self.power_variation),
            'room_power': dict(self.room_power),
            'devices_by_room': self.devices_by_room
        }

    def generate_power_consumption_response(self, user_message):
        """
        Generate responses about power consumption
        """
        user_message = user_message.lower()
        
        try:
            # Total power consumption
            if "total power consumption" in user_message or "total consumption" in user_message:
                return f"The total power consumption across all devices is {self.total_power:.2f} power units."
            
            # Kitchen percentage
            if "percentage" in user_message and "kitchen" in user_message:
                kitchen_power = self.room_power.get('kitchen', 0)
                percentage = (kitchen_power / self.total_power) * 100
                return f"Kitchen appliances consume {percentage:.2f}% of total power ({kitchen_power:.2f} out of {self.total_power:.2f} power units)."
            
            # Variable power consumption
            if "variable" in user_message or "variation" in user_message:
                most_variable = self.power_variation.idxmax()
                variation = self.power_variation[most_variable]
                return f"{most_variable} shows the most variable power consumption with a variation of {variation:.2f} units between its minimum ({self.min_power[most_variable]:.2f}) and maximum ({self.peak_power[most_variable]:.2f}) power consumption."
            
            # Most energy-efficient
            if "efficient" in user_message or "least power" in user_message:
                device = self.sorted_appliances.index[-1]
                power = self.sorted_appliances.iloc[-1]
                return f"The most energy-efficient appliance is {device} with {power:.2f} total average power units."
            
            # Highest peak power
            if "peak power" in user_message or "highest peak" in user_message:
                device = self.peak_power.idxmax()
                peak = self.peak_power[device]
                return f"{device} has the highest peak power consumption at {peak:.2f} power units."
            
            # Devices in room
            if "devices" in user_message and "in" in user_message:
                for room in self.devices_by_room.keys():
                    if room.lower() in user_message:
                        devices = self.devices_by_room[room]
                        return f"The {room} contains the following devices: {', '.join(devices)}."
                    
            # Most power consuming device
            if any(phrase in user_message for phrase in ['most power', 'highest consumption', 'most electricity']):
                top_device = self.sorted_appliances.index[0]
                top_power = self.sorted_appliances.iloc[0]
                return f"{top_device} consumes the most power with {top_power:.2f} total average power units."
            
            # Least power consuming device
            if any(phrase in user_message for phrase in ['least power', 'lowest consumption']):
                bottom_device = self.sorted_appliances.index[-1]
                bottom_power = self.sorted_appliances.iloc[-1]
                return f"{bottom_device} consumes the least power with {bottom_power:.2f} total average power units."
            
            # Comparison between devices
            if 'compare' in user_message or 'between' in user_message:
                # Extract potential device names from the dataset
                available_devices = [device.lower() for device in self.sorted_appliances.index]
                
                # Find matching devices in the user message
                matched_devices = [
                    device for device in available_devices 
                    if device in user_message
                ]
                
                # If exactly two devices are found
                if len(matched_devices) == 2:
                    # Get the actual device names from the original index
                    dev1 = self.sorted_appliances.index[self.sorted_appliances.index.str.lower() == matched_devices[0]][0]
                    dev2 = self.sorted_appliances.index[self.sorted_appliances.index.str.lower() == matched_devices[1]][0]
                    
                    power1 = self.total_power_consumption[dev1]
                    power2 = self.total_power_consumption[dev2]
                    
                    # Detailed comparison
                    if power1 > power2:
                        difference = ((power1 - power2) / power2) * 100
                        return (f"{dev1} consumes more power than {dev2}. "
                                f"{dev1}: {power1:.2f}, {dev2}: {power2:.2f} total average power units. "
                                f"{dev1} consumes {difference:.2f}% more power.")
                    else:
                        difference = ((power2 - power1) / power1) * 100
                        return (f"{dev2} consumes more power than {dev1}. "
                                f"{dev2}: {power2:.2f}, {dev1}: {power1:.2f} total average power units. "
                                f"{dev2} consumes {difference:.2f}% more power.")
            
            # Room-based queries
            if 'room' in user_message:
                # Extract room from the message
                rooms = self.df['room'].unique()
                matched_rooms = [room for room in rooms if room.lower() in user_message]
                
                if matched_rooms:
                    room = matched_rooms[0]
                    room_devices = self.df[self.df['room'] == room]
                    room_power = room_devices.groupby('device')['avg_power'].sum().sort_values(ascending=False)
                    
                    response_parts = [f"Power consumption in {room}:"]
                    for device, power in room_power.items():
                        response_parts.append(f"{device}: {power:.2f} average power units")
                    
                    return " ".join(response_parts)
            
            # Fallback to QA pipeline
            result = self.qa_pipeline({
                'context': str(self.context),
                'question': user_message
            })
            
            if result['score'] >= 0.1:
                return result['answer']
        
        except Exception as e:
            print(f"Error in generating response: {e}")
            return "I encountered an error processing your question. Please try rephrasing it."
        
        # Generic response if no specific match
        return "Sorry, I couldn't find a specific answer to your question about power consumption."

    def prepare_train_features(self, examples):
        """
        Prepare features for model training
        """
        tokenizer_kwargs = {
            'truncation': True, 
            'max_length': 384, 
            'stride': 128, 
            'return_overflowing_tokens': True,
            'padding': 'max_length'
        }
        
        tokenized_examples = self.tokenizer(
            examples['question'],
            examples['context'],
            **tokenizer_kwargs
        )
        
        sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
        offset_mapping = tokenized_examples.pop("offset_mapping")
        
        tokenized_examples["start_positions"] = []
        tokenized_examples["end_positions"] = []
        
        for i, offsets in enumerate(offset_mapping):
            sample_idx = sample_mapping[i]
            answer = examples['answer'][sample_idx]
            
            start_char = answer['answer_start']
            end_char = start_char + len(answer['text'])
            
            # Find token indices for answer
            sequence_ids = tokenized_examples.sequence_ids(i)
            
            # Determine token start and end
            token_start_index = 0
            while sequence_ids[token_start_index] != 1:
                token_start_index += 1
            
            token_end_index = len(sequence_ids) - 1
            while sequence_ids[token_end_index] != 1:
                token_end_index -= 1
            
            # Check if answer is within this example
            if not (offsets[token_start_index][0] <= start_char and 
                    offsets[token_end_index][1] >= end_char):
                tokenized_examples["start_positions"].append(0)
                tokenized_examples["end_positions"].append(0)
            else:
                while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
                    token_start_index += 1
                tokenized_examples["start_positions"].append(token_start_index - 1)
                
                while token_end_index > 0 and offsets[token_end_index][1] >= end_char:
                    token_end_index -= 1
                tokenized_examples["end_positions"].append(token_end_index + 1)
        
        return tokenized_examples

def fine_tune_model(self, custom_questions=None, output_dir='./results', epochs=3):
    """
    Fine-tune the model with power consumption specific questions
    """
    # If no custom questions provided, generate comprehensive questions based on the dataset
    if custom_questions is None:
        # Calculate some additional metrics for questions
        room_power = self.df.groupby('room')['avg_power'].agg(['sum', 'mean'])
        peak_power = self.df.groupby('device')['max_power'].max()
        min_power = self.df.groupby('device')['min_power'].min()
        time_analysis = self.df.groupby(['device', pd.to_datetime(self.df['start_time']).dt.hour])['avg_power'].mean()

        custom_questions = [
            # Basic consumption questions
            {
                'question': 'Which device consumes the most power?',
                'context': self.context,
                'answer': f"{self.sorted_appliances.index[0]} with {self.sorted_appliances.iloc[0]:.2f} average power units"
            },
            {
                'question': 'What is the total power consumption of all devices?',
                'context': self.context,
                'answer': f"Total power consumption is {self.total_power_consumption.sum():.2f} average power units"
            },
            
            # Room-based questions
            {
                'question': 'Which room has the highest average power consumption?',
                'context': f"Room power consumption: {', '.join([f'{room}: {power:.2f}' for room, power in room_power['mean'].items()])}",
                'answer': f"{room_power['mean'].idxmax()} with {room_power['mean'].max():.2f} average power units"
            },
            {
                'question': 'What devices are in the kitchen?',
                'context': f"Devices by room: {', '.join([f'{room}: {list(self.df[self.df.room == room].device.unique())}' for room in self.df.room.unique()])}",
                'answer': f"The kitchen contains: {list(self.df[self.df.room == 'kitchen'].device.unique())}"
            },
            
            # Peak power questions
            {
                'question': 'Which device has the highest peak power consumption?',
                'context': f"Peak power consumption by device: {', '.join([f'{dev}: {power:.2f}' for dev, power in peak_power.items()])}",
                'answer': f"{peak_power.idxmax()} with a peak consumption of {peak_power.max():.2f} power units"
            },
            {
                'question': 'What is the difference between maximum and minimum power for the refrigerator?',
                'context': f"Power range for refrigerator - Max: {peak_power.get('refrigerator', 0):.2f}, Min: {min_power.get('refrigerator', 0):.2f}",
                'answer': f"The difference is {(peak_power.get('refrigerator', 0) - min_power.get('refrigerator', 0)):.2f} power units"
            },
            
            # Comparative questions
            {
                'question': 'Compare power consumption of kitchen and living room',
                'context': f"Room total consumption: Kitchen: {room_power.loc['kitchen', 'sum']:.2f}, Living Room: {room_power.loc['living room', 'sum']:.2f}",
                'answer': f"Kitchen consumes {room_power.loc['kitchen', 'sum']:.2f} units while Living Room consumes {room_power.loc['living room', 'sum']:.2f} units"
            },
            {
                'question': 'What is the most energy-efficient appliance?',
                'context': self.context,
                'answer': f"{self.sorted_appliances.index[-1]} with {self.sorted_appliances.iloc[-1]:.2f} average power units"
            },
            
            # Time-based questions
            {
                'question': 'When does the air conditioner consume the most power?',
                'context': f"Air conditioner hourly consumption: {time_analysis['air conditioner'].to_dict()}",
                'answer': f"The air conditioner peaks at hour {time_analysis['air conditioner'].idxmax()[1]}"
            },
            
            # Complex analysis questions
            {
                'question': 'What percentage of total power is consumed by kitchen appliances?',
                'context': f"Kitchen total: {room_power.loc['kitchen', 'sum']:.2f}, Overall total: {self.total_power_consumption.sum():.2f}",
                'answer': f"Kitchen appliances consume {(room_power.loc['kitchen', 'sum'] / self.total_power_consumption.sum() * 100):.2f}% of total power"
            },
            {
                'question': 'Which devices show the most variable power consumption?',
                'context': f"Power variation by device: {', '.join([f'{dev}: {peak_power[dev] - min_power[dev]:.2f}' for dev in peak_power.index])}",
                'answer': f"{(peak_power - min_power).idxmax()} shows the most variation with {(peak_power - min_power).max():.2f} units difference"
            }
        ]