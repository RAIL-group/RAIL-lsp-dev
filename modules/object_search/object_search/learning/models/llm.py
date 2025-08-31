import os
import re
from openai import OpenAI
from google import genai
from pathlib import Path
from dotenv import load_dotenv
from collections import Counter
import hashlib

load_dotenv(dotenv_path='/data/.env')


class LLM:
    """Abstract LLM class"""
    def __init__(self, client=None, model_name="", prompt_cache_dir=None):
        self.client = client
        self.model_name = model_name
        if prompt_cache_dir is None:
            self.prompt_cache_path = None
            return
        self.prompt_cache_path = Path(prompt_cache_dir) / f"{self.model_name}_cache"
        self.prompt_cache_path.mkdir(parents=True, exist_ok=True)

    def query_llm(self, prompt):
        raise NotImplementedError

    def get_response(self, prompt):
        cached_response = self.get_cached_response(prompt)
        if cached_response is not None:
            return cached_response
        response = self.query_llm(prompt)
        self.save_response_to_cache(prompt, response)
        return response

    def get_cached_response(self, prompt):
        if self.prompt_cache_path is None:
            return None
        prompt_md5 = hashlib.md5(prompt.encode()).hexdigest()
        cache_file = self.prompt_cache_path / f"{prompt_md5}.txt"
        if not cache_file.exists():
            return None
        with open(cache_file) as f:
            lines = f.read().splitlines()
        if len(lines) != 2:
            print(f"Cache in {cache_file.name} is not valid.")
            return None
        print(f"Using cached {self.model_name} response from {cache_file.name}")
        return lines[-1]

    def save_response_to_cache(self, prompt, response):
        if self.prompt_cache_path is None:
            return
        prompt_md5 = hashlib.md5(prompt.encode()).hexdigest()
        cache_file = self.prompt_cache_path / f"{prompt_md5}.txt"
        with open(cache_file, "w") as f:
            f.write(prompt.strip() + "\n" + response.strip() + "\n")

    @classmethod
    def get_net_eval_fn(cls,
                        prompt_template_id=0,
                        fake_llm_response_text=None,
                        prompt_cache_path=None):
        llm_model = cls(prompt_cache_path=prompt_cache_path)

        def get_properties(datum, subgoals):
            prob_feasible_dict = {}
            if fake_llm_response_text is not None:
                print("******Using fake LLM response******")
                for subgoal in subgoals:
                    prob_feasible_dict[subgoal] = parse_llm_response(fake_llm_response_text)
                return prob_feasible_dict

            graph = datum['graph']
            target_object_name = datum['target_obj_info']['name']
            for subgoal in subgoals:
                subgoal_container_name = graph.get_node_name_by_idx(subgoal.id)
                parent_node_idx = graph.get_parent_node_idx(subgoal.id)
                room_name = graph.get_node_name_by_idx(parent_node_idx)
                prompt = generate_prompt(graph,
                                         target_object_name,
                                         subgoal_container_name,
                                         room_name,
                                         prompt_template_id)
                response = llm_model.get_response(prompt)
                prob_feasible_dict[subgoal] = parse_llm_response(response)
            return prob_feasible_dict

        return get_properties

    @classmethod
    def get_search_action_fn(cls, prompt_template_id=0, prompt_cache_path=None):
        llm_model = cls(prompt_cache_path=prompt_cache_path)

        def get_search_action(datum):
            graph = datum['graph']
            subgoals = datum['subgoals']
            target_object_name = datum['target_obj_info']['name']
            room_distances = datum['room_distances']
            robot_distances = datum['robot_distances']
            prompt, subgoal_description_to_idx = generate_prompt_llm_as_planner(graph,
                                                                                target_object_name,
                                                                                subgoals,
                                                                                room_distances,
                                                                                robot_distances,
                                                                                prompt_template_id)
            response = llm_model.get_response(prompt)
            chosen_subgoal = identify_subgoal_from_response(response, subgoal_description_to_idx)
            if chosen_subgoal is None:
                print("Using nearest subgoal as chosen subgoal.")
                chosen_subgoal = min(subgoals, key=robot_distances.get)
            return chosen_subgoal

        return get_search_action


class GPT(LLM):
    def __init__(self, model_name="gpt-5-mini", prompt_cache_path=None):
        client = OpenAI(organization='org-rjqQxMqdPyu0NpSFXshcA88Q',
                        project='proj_5gbQGApHd9F8blfA4mpiKE9R',
                        api_key=os.getenv('OPENAI_API_KEY'))
        super().__init__(client, model_name, prompt_cache_path)

    def query_llm(self, prompt):
        print(f"Querying GPT model {self.model_name}")
        result = self.client.responses.create(
            model=self.model_name,
            input=prompt
        )
        response = result.output_text.strip()
        return response


class Gemini(LLM):
    def __init__(self, model_name="gemini-2.5-flash", prompt_cache_path=None):
        client = genai.Client(api_key=os.getenv('GEMINI_API_KEY'))
        super().__init__(client, model_name, prompt_cache_path)

    def query_llm(self, prompt):
        print(f"Querying Gemini model {self.model_name}")
        result = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt
        )
        response = result.text.strip()
        return response


def generate_prompt(graph, target_object_name, subgoal_container_name, room_name, prompt_template_id):
    """Generate the prompt for the LLM model."""
    description = generate_description(graph)
    target_object_name = clean_name(target_object_name)
    subgoal_container_name = clean_name(subgoal_container_name)
    room_name = clean_name(room_name)

    templates = {
        'prompt_a': (
            f"You are serving as part of a system in which a robot needs to find objects located around a household. "
            f"Here is a schema that describes the connectivity of rooms in the house: {description} "
            "You will be asked to estimate the probability (a value between 1% and 100%) of where "
            "objects are located in that house, "
            "leveraging your considerable experience in how human occupied spaces are located. "
            "You must produce a numerical value and nothing else, "
            "as it is important to the overall functioning of the system. "
            "Here is an example exchange for an arbitrary house: "
            "User: 'What is the likelihood that I find eggs in the refrigerator in the kitchen?' You: '90%'. "
            "The logic here is that there is a high likelihood that a typical refrigerator in the "
            "kitchen contains eggs, but it is not guaranteed as not all refrigerators have eggs. "
            "Here is your prompt for today: "
            f"What is the likelihood that I find {target_object_name} in the "
            f"{subgoal_container_name} in the {room_name}?"
        ),
        'prompt_b': (
            f"You are assisting in a robotic system designed to locate items within a residence. "
            f"The following is a description of the layout and connectivity between rooms in the home: {description} "
            "Your task is to estimate the likelihood (a percentage from 1% to 100%) that a specified object "
            "is in a given location. "
            "Base your reasoning on general patterns of human behavior and usage of household spaces. "
            "Your response must be a single numerical value, with no additional explanation, as precision "
            "is critical to system operation. "
            "Example exchange: "
            "User: 'What is the probability of finding bread in the pantry in the kitchen?' You: '85%'. "
            "The reasoning here is that bread is commonly stored in pantries, but exceptions exist, such as "
            "if it is refrigerated. "
            "Now, respond to this prompt: "
            f"What is the probability of finding {target_object_name} in the "
            f"{subgoal_container_name} in the {room_name}?"
        ),
        'prompt_minimal': (
            f"What is the probability of finding {target_object_name} in the "
            f"{subgoal_container_name} in the {room_name} of a typical household? "
            "Your response should only include a numerical percentage value between 1% to 100% and nothing else."
        )
    }

    prompt = templates.get(prompt_template_id)
    if prompt is None:
        raise ValueError(f"{prompt_template_id=} is not valid. Valid ids are: {list(templates.keys())}")
    return prompt


def parse_llm_response(response):
    """Parse the response from the LLM model and extract a normalized probability value."""
    match = re.search(r'(\d+(\.\d+)?)%', response)
    if match:
        percentage_value = float(match.group(1))
        return max(percentage_value / 100.0, 0.01)
    try:
        numeric_value = float(response.strip())
        return max(numeric_value / 100.0, 0.01)
    except ValueError:
        print(f'Failed to parse response: "{response}". Using 0.01 as probability value.')
        return 0.01


def generate_description(graph):
    """Generate a natural language description of the graph, including the type of space, rooms, and their contents."""
    nodes = graph.nodes
    edges = graph.edges

    # Initialize dictionaries to store room contents and room names
    room_contents = {}
    room_names = []

    # Extract the type of the space (e.g., apartment)
    space_type = "Unknown"
    for node in nodes.values():
        if node['type'][0] == 1:  # Check for the space node (e.g., apartment)
            space_type = clean_name(node['name'])
            break

    # Traverse the edges to find containers in rooms (room -> container connections)
    for parent, child in edges:
        parent_node = nodes[parent]
        child_node = nodes[child]

        # Check if the parent is a room and the child is a container
        if parent_node['type'] == [0, 1, 0, 0] and child_node['type'] == [0, 0, 1, 0]:
            room_name = clean_name(parent_node['name'])
            if room_name not in room_contents:
                room_contents[room_name] = []
            room_contents[room_name].append(clean_name(child_node['name']))
            if room_name not in room_names:
                room_names.append(room_name)

    # Generate a natural language description
    descriptions = []
    rooms_description = f"The {space_type} contains the following rooms: {', '.join(sorted(room_names))}."
    descriptions.append(rooms_description)

    # Add description for contents of each room
    for room, contents in room_contents.items():
        description = f"The {room} contains: {', '.join(sorted(set(contents)))}."
        descriptions.append(description)

    return " ".join(descriptions)


def clean_name(name):
    """Clean object names according to the mapping."""
    name_mapping = {
        'diningtable': 'dining table',
        'shelvingunit': 'shelving unit',
        'garbagebag': 'garbage bag',
        'tabletopdecor': 'tabletop decor',
        'basketball': 'basket ball',
        'baseballbat': 'baseball bat',
        'garbagecan': 'garbage can',
        'livingroom': 'living room',
        'coffeetable': 'coffee table',
        'endtable': 'end table',
        'kitchencounter': 'kitchen counter',
        'kitchencabinet': 'kitchen cabinet',
        'bathroomcabinet': 'bathroom cabinet',
        'showerstall': 'shower stall',
        'bathtubbasin': 'bathtub basin',
        'bathroomcounter': 'bathroom counter',
        'tvstand': 'tv stand',
        'nightstand': 'night stand',
        'closetshelf': 'closet shelf',
        'laptops': 'laptop',
        'desklamp': 'desk lamp',
        'floorlamp': 'floor lamp',
        'walllamp': 'wall lamp',
        'tablelamp': 'table lamp',
    }
    return name_mapping.get(name, name)


def generate_prompt_llm_as_planner(graph, target_object_name, subgoals,
                                   room_distances, robot_distances, prompt_template_id):
    """Generate the prompt for the LLM model to directly get the container to search."""
    room_idx_to_name = get_room_idx_to_unique_name_mapping(graph)
    subgoal_description_to_idx = get_subgoal_description_to_idx_mapping(graph, subgoals, room_idx_to_name)
    room_names = list(room_idx_to_name.values())
    subgoal_descriptions = list(subgoal_description_to_idx.keys())
    subgoal_descriptions = sorted(subgoal_descriptions, key=lambda x: room_names.index(x.split(' in ')[1]))
    robot_location = get_robot_location_room_name(graph, robot_distances, room_idx_to_name)
    room_distances_description = get_room_distances_description(graph, room_distances, room_idx_to_name)
    search_locations_description = get_search_locations_description(subgoal_descriptions)

    templates = {
        'prompt_direct': (
            "You are assisting a robot in locating objects within a household based on a provided map of rooms "
            "and their contents. "
            "Your task is to determine the exact location where the specified object can be found, "
            "based on given description of the household. "
            "You will be asked pick a location to visit where the object could be found quickly. "
            "You should only pick one location from the given list. "
            "Here is an example: "
            "User: The apartment contains: bathroom, bedroom, kitchen. The distance between rooms is as follows: "
            "bathroom and bedroom: 5.95 meters, bedroom and kitchen: 3.25 meters, bathroom and kitchen: 4.75 meters. "
            "The robot is currently located at bathroom and is looking for pillow. "
            "Available locations to search are: sink in bathroom, toilet in bathroom, bed in bedroom, "
            "sidetable in bedroom. "
            "Which of the given search locations should the robot visit to quickly find pillow? "
            "You: bed in bedroom. "
            f"Now give your answer for another household with the following layout: {room_distances_description} "
            f"The robot is currently located at {robot_location} and is looking for {target_object_name}. "
            f"Available locations to search are: {search_locations_description}. "
            f"Which of the given search locations should the robot visit to quickly find {target_object_name}? "
            f"Respond with a search location and nothing else."
        )
    }
    if prompt_template_id not in templates:
        raise ValueError(f"{prompt_template_id=} is not valid. Valid ids are: {list(templates.keys())}")
    return templates[prompt_template_id], subgoal_description_to_idx


def identify_subgoal_from_response(response, subgoal_description_to_idx):
    """Parse the response from LLM to map it to the subgoal."""
    response = response.strip('.')
    subgoal = subgoal_description_to_idx.get(response)
    if subgoal is not None:
        print(f"Chosen container: {response}")
        return subgoal
    print('Matching approximately to subgoal ...')
    for k, v in subgoal_description_to_idx.items():
        if k in response:
            print(f"Chosen container: {response}")
            return v

    print(f"Failed to map response to any subgoals:\n{response=}")
    print("Available subgoals:\n")
    print('\n'.join(subgoal_description_to_idx.keys()))


def get_room_idx_to_unique_name_mapping(graph):
    room_indices = graph.room_indices
    room_names = [clean_name(graph.get_node_name_by_idx(room_idx)) for room_idx in room_indices]
    room_names = rename_duplicate_names(room_names)
    room_idx_to_name = {room_idx: room_name for room_idx, room_name in zip(room_indices, room_names)}
    return room_idx_to_name


def get_subgoal_description_to_idx_mapping(graph, subgoals, room_idx_to_name):
    container_names = [clean_name(graph.get_node_name_by_idx(subgoal.id)) for subgoal in subgoals]
    container_room_idx = [graph.get_parent_node_idx(subgoal.id) for subgoal in subgoals]
    container_room_names = [room_idx_to_name[room_idx] for room_idx in container_room_idx]
    container_room_pairs = list(zip(container_names, container_room_names))
    container_room_pairs = rename_duplicate_items(container_room_pairs)
    container_room_pairs_to_subgoal_idx = {f'{container} in {room}': subgoals[i]
                                           for i, (container, room) in enumerate(container_room_pairs)}
    return container_room_pairs_to_subgoal_idx


def get_room_distances_description(graph, room_distances, room_idx_to_name):
    """Generate a natural language description about rooms and distances between them."""
    nodes = graph.nodes
    # Extract the type of the space (e.g., apartment)
    space_type = "Unknown"
    for node in nodes.values():
        if node['type'][0] == 1:  # Check for the space node (e.g., apartment)
            space_type = clean_name(node['name'])
            break

    descriptions = []
    for (room1_idx, room2_idx), distance in room_distances.items():
        room1_name = room_idx_to_name[room1_idx]
        room2_name = room_idx_to_name[room2_idx]
        descriptions.append(f"{room1_name} and {room2_name}: {distance:.1f} meters")

    description = f"The {space_type} contains the following rooms: {', '.join(sorted(room_idx_to_name.values()))}. "
    if len(descriptions) > 0:
        description += f"The distance between rooms is as follows: {', '.join(descriptions)}."

    return description


def get_search_locations_description(subgoal_descriptions):
    """Generate a natural language description of the search locations."""
    subgoal_descriptions = sorted(subgoal_descriptions, key=lambda x: x.split(' in ')[1])
    description = ', '.join(subgoal_descriptions)
    return description


def get_robot_location_room_name(graph, robot_distances, room_idx_to_name):
    """Get the room name where the robot is located based on the nearest container."""
    nearest_container_idx = min(robot_distances, key=robot_distances.get)
    room_node_idx = graph.get_parent_node_idx(nearest_container_idx)
    room_name = room_idx_to_name[room_node_idx]
    return room_name


def rename_duplicate_names(room_names):
    occurrences = Counter(room_names)
    counts = {}
    updated_room_names = []
    for s in room_names:
        if occurrences[s] > 1:  # Only rename duplicates
            counts[s] = counts.get(s, 0) + 1
            updated_room_names.append(f"{s} {counts[s]}")
        else:
            updated_room_names.append(s)

    return updated_room_names


def rename_duplicate_items(container_room_pairs):
    occurrences = Counter(container_room_pairs)
    counts = {}
    updated_pairs = []
    for container, room in container_room_pairs:
        key = (container, room)
        if occurrences[key] > 1:  # Only rename duplicates
            counts[key] = counts.get(key, 0) + 1
            updated_pairs.append((f"{container} {counts[key]}", room))
        else:
            updated_pairs.append((container, room))

    return updated_pairs
