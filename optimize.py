import argparse
import requests
import json
from PIL import Image
import urllib
import time
from io import BytesIO
from typing import List
import pandas as pd
from tqdm.auto import tqdm
import uuid
import boto3
from botocore.exceptions import ClientError
import datetime
import os
import sys
import time
import logging
import numpy as np
import toloka.client as toloka
from tqdm.auto import trange
from crowdkit.aggregation import BradleyTerry
from collections import Counter
from tqdm.auto import tqdm, trange
import scipy.stats as sps
import pickle


class DiffusionApi:
    """
    An example API class. You can use any class with generate function returning PIL image.
    But you need to modify ImageGenerator's code a bit.
    """
    def __init__(self, base_url):
        self.base_url = base_url
        
    def generate(self, prompt, steps=50, scale=7.5, seed=0, height=512, width=512, mode='none', faces=False, upscale=False):
        values = {
            'prompt': prompt, 'steps': steps,
            'scale': str(scale), 'mode': mode, 'upscale': 'no' if not upscale else 'yes',
            'faces': 'no' if not faces else 'yes',
            'seed': seed, 'height': height, 'width': width
        }

        r = requests.post(f'{self.base_url}/generate', data=values)
        job = json.loads(r.content.decode('utf-8'))
        uid = job['uid']

        status = job['status']

        while status != 'completed':
            r = requests.get(f'{self.base_url}/task_status?' + urllib.parse.urlencode({'uid': uid}))
            job = json.loads(r.content.decode('utf-8'))
            status = job['status']
            time.sleep(0.1)

        r = requests.get(f'{self.base_url}/get_image?' + urllib.parse.urlencode({'uid': uid}))
        return Image.open(BytesIO(r.content)).convert("RGB")


class ImageGenerator:
    def __init__(self, api):
        self.api = api
        
    def generate(self, prompt: str, keywords: List[str], orientation: str = 'square', n_images: int = 4):
        final_prompt = ', '.join([prompt] + keywords)
        
        if orientation == 'square':
            height = 512
            width = 512
        elif orientation == 'album':
            height = 512
            width = 768
        elif orientation == 'portrait':
            height = 768
            width = 512
            
        images = []
        for seed in range(n_images):
            images.append(self.api.generate(final_prompt, seed=seed, height=height, width=width))
        return images
    
    def generate_bucket(self, prompts: List[str], orientations: List[str], keywords: List[str], n_images: int = 4):
        bucket_generations = []
        for prompt, orientation in tqdm(zip(prompts, orientations), total=len(prompts)):
            bucket_generations.append(self.generate(prompt, sorted(keywords), orientation, n_images))
        return bucket_generations


class ImageSaver:
    def __init__(
        self, 
        aws_access_key_id,
        aws_secret_access_key,
        endpoint_url,
        bucket
    ):
        self.aws_access_key_id = aws_access_key_id
        self.aws_secret_access_key = aws_secret_access_key
        self.endpoint_url = endpoint_url
        self.bucket = bucket
        
    def upload_file(self, file_name, object_name=None):
        if object_name is None:
            object_name = os.path.basename(file_name)

        session = boto3.session.Session()

        s3_client = session.client(
            service_name='s3',
            aws_access_key_id=self.aws_access_key_id,
            aws_secret_access_key=self.aws_secret_access_key,
            endpoint_url=self.endpoint_url,
        )
        try:
            response = s3_client.upload_file(file_name, self.bucket, object_name)
        except ClientError as e:
            logging.error(e)
            return False
        return True
    
    def save_image_batch(self, image_batch):
        batch_name = uuid.uuid4().hex
        urls = []
        
        for i, image in enumerate(image_batch):
            filename = f'{batch_name}_{i}.png'
            image.save(f'gen_images/{filename}')
            self.upload_file(f'gen_images/{filename}', filename)
            urls.append(f'{self.endpoint_url}/{self.bucket}/{filename}')
        return urls
    
    def save_images(self, images):
        urls = []
        for batch in tqdm(images):
            urls.append(self.save_image_batch(batch))
        return urls


def generate_tasks(buckets: List[List[List[str]]]):
    n_prompts = len(buckets[0])
    tasks = [[] for _ in range(n_prompts)]
    for i in range(n_prompts):
        for j in range(len(buckets)):
            tasks[i].append(buckets[j][i])
    return tasks


def wait_pool_for_close(pool_id, minutes_to_wait=1):
    sleep_time = 60 * minutes_to_wait
    pool = toloka_client.get_pool(pool_id)
    while not pool.is_closed():
        op = toloka_client.get_analytics([toloka.analytics_request.CompletionPercentagePoolAnalytics(subject_id=pool.id)])
        op = toloka_client.wait_operation(op)
        percentage = op.details['value'][0]['result']['value']
        logging.info(
            f'   {datetime.datetime.now().strftime("%H:%M:%S")}\t'
            f'Pool {pool.id} - {percentage}%'
        )
        time.sleep(sleep_time)
        pool = toloka_client.get_pool(pool.id)
    logging.info('Pool was closed.')


class TaskProcessor:
    def __init__(self, toloka_client, base_pool_id: int, prompts: List[str], honeypots: List[List[str]], k: int = 3):
        self.toloka_client = toloka_client
        self.base_pool_id = base_pool_id
        self.honeypots = honeypots
        self.prompts = prompts
        self.k = k
        self.results = None
        self.existing_tasks = []
        self.prompt2id = {prompt: i for i, prompt in enumerate(self.prompts)}
        
    def generate_inside_batch_pair(self, n_items):
        l = 0
        r = 0
        while l == r:
            l = np.random.randint(0, n_items)
            r = np.random.randint(0, n_items)
        return l, r
    
    def generate_inside_batch_pairs(self, n_items):
        pairs = []
        n_pairs = int(self.k * np.log2(n_items))
        
        for _ in range(n_pairs):
            pairs.append(self.generate_inside_batch_pair(n_items))
        return pairs
            
    def make_task_from_pair(self, pool_id, prompt, prompt_tasks, pair):
        l, r = pair
        input_values = {}
        for i in range(4):
            # print(prompt_tasks)
            # print(l, i)
            input_values[f'left_{i}'] = prompt_tasks[l][i]
        for i in range(4):
            input_values[f'right_{i}'] = prompt_tasks[r][i]
        
        input_values['prompt'] = prompt
        
        return toloka.Task(
            pool_id=pool_id,
            input_values=input_values,
        )
    
    
    def make_tasks_from_pairs(self, pool_id, prompt, prompt_tasks, pairs):
        return [self.make_task_from_pair(pool_id, prompt, prompt_tasks, pair) for pair in pairs]
    
    
    def make_golden_tasks(self, pool_id):
        tasks = []
        for _, row in self.honeypots.iterrows():
            input_values = {}
            input_values['prompt'] = row['prompt']
            for i in range(4):
                input_values[f'left_{i}'] = row[f'left_{i}']
                input_values[f'right_{i}'] = row[f'right_{i}']
            tasks.append(
                toloka.Task(
                    pool_id=pool_id,
                    input_values=input_values,
                    known_solutions = [
                        toloka.task.BaseTask.KnownSolution(
                            output_values={'result': row['gt']}
                        )
                    ],
                    infinite_overlap=True,
                )
            )
        return tasks
        
        
    def add_init_tasks(self, pre_tasks):
        self.existing_tasks = pre_tasks
        pool = self.toloka_client.clone_pool(pool_id=self.base_pool_id)
        self.toloka_client.create_tasks(self.make_golden_tasks(pool.id), allow_defaults=True)

        for prompt_id in trange(len(self.prompts)):
            pairs = self.generate_inside_batch_pairs(len(pre_tasks[prompt_id]))
            tasks = self.make_tasks_from_pairs(pool.id, self.prompts[prompt_id], pre_tasks[prompt_id], pairs)
            self.toloka_client.create_tasks(tasks, allow_defaults=True)
            
        pool = self.toloka_client.open_pool(pool.id)
        wait_pool_for_close(pool.id)
        return self.process_results(pool.id)
        
        
    def add_task_to_existing(self, task):
        for i in range(len(task)):
            self.existing_tasks[i].append(task[i])
            
    def generate_new_pair(self):
        r = len(self.existing_tasks[0]) - 1
        l = np.random.randint(0, len(self.existing_tasks[0]) - 1)
        if np.random.randint(0, 2) == 0:
            return l, r
        else:
            return r, l
            
    def generate_new_pairs(self):
        n = len(self.existing_tasks[0]) - 1
        n_pairs = int(self.k * ((n + 1) * np.log2(n + 1) - n * np.log(n)))
        return [self.generate_new_pair() for i in range(n_pairs)]
        

    def add_task(self, task):
        pool = self.toloka_client.clone_pool(pool_id=self.base_pool_id)
        self.add_task_to_existing(task)
        self.toloka_client.create_tasks(self.make_golden_tasks(pool.id), allow_defaults=True)
        for prompt_id in trange(len(self.prompts)):
            pairs = self.generate_new_pairs()
            # print(pairs)
            tasks = self.make_tasks_from_pairs(pool.id, self.prompts[prompt_id], self.existing_tasks[prompt_id], pairs)
            self.toloka_client.create_tasks(tasks, allow_defaults=True)
        
        pool = self.toloka_client.open_pool(pool.id)
        wait_pool_for_close(pool.id)
        return self.process_results(pool.id)
        
    def process_results(self, pool_id):
        answers_df = self.toloka_client.get_assignments_df(pool_id)
        results = [[] for _ in range(len(self.prompts))]
        for i, row in answers_df.iterrows():
            left_uid = row['INPUT:left_0'].split('_')[0].split('/')[-1]
            right_uid = row['INPUT:right_0'].split('_')[0].split('/')[-1]
            prompt_id = self.prompt2id[row['INPUT:prompt']]
            worker_id = row['ASSIGNMENT:worker_id']
            out = row['OUTPUT:result']
            results[prompt_id].append([left_uid, right_uid, worker_id, out])
        return results


class Evaluator:
    def __init__(self, prompts, orientations, keywords, generator, img_saver, task_processor):
        self.prompts = prompts
        self.orientations = orientations
        self.keywords = np.array([k for k in keywords])
        self.generator = generator
        self.img_saver = img_saver
        self.task_processor = task_processor
        
        self.population = []
        self.values = []
        self.annotation = [[] for _ in range(len(self.prompts))]
        self.uid2mask = {}
        
    def mask2keywords(self, mask):
        return self.keywords[mask]
    
    def get_uid(self, url):
        return url.split('_')[0].split('/')[-1]
    
    def set_uids(self, urls, idx):
        for u in urls:
            uid = self.get_uid(u[0])
            self.uid2mask[uid] = idx
            
    def process_results(self):
        positions = Counter()
        for prompt_annotation in tqdm(self.annotation):
            data = pd.DataFrame(
                [[x[0], x[1], x[2], x[0] if x[3] == 'left' else x[1]] for x in prompt_annotation if x[0] in self.uid2mask and x[1] in self.uid2mask],
                columns=['left', 'right', 'worker', 'label']
            )
            
            agg_res = BradleyTerry(500).fit_predict(data)
            agg_res = agg_res.sort_values()
            for i, (uid, score) in enumerate(agg_res.iteritems()):
                positions[self.uid2mask[uid]] += i
        
        for i, avg_pos in positions.items():
            self.values[i] = avg_pos / len(self.prompts)
        
        return self.values
        
    
    def initialize(self, masks):
        buckets = []
        for mask in masks:
            keywords = self.mask2keywords(mask)
            gen_img = self.generator.generate_bucket(self.prompts, self.orientations, keywords)
            urls = self.img_saver.save_images(gen_img)
            self.population.append(mask)
            self.values.append(0)
            self.set_uids(urls, len(self.population) - 1)
        
            buckets.append(urls)
        tasks = generate_tasks(buckets)
        results = self.task_processor.add_init_tasks(tasks)
        for i in range(len(results)):
            self.annotation[i] += results[i]
        return self.process_results()
    
    def add_candidate(self, mask):
        if np.sum(mask) > 15:
            self.population.append(mask)
            self.values.append(-np.sum(mask))
            return self.values
            
        keywords = self.mask2keywords(mask)
        gen_img = self.generator.generate_bucket(self.prompts, self.orientations, keywords)
        urls = self.img_saver.save_images(gen_img)
        self.population.append(mask)
        self.values.append(0)
        self.set_uids(urls, len(self.population) - 1)
        results = self.task_processor.add_task(urls)
        for i in range(len(results)):
            self.annotation[i] += results[i]
        return self.process_results()


class GeneticOpt:
    def __init__(self, evaluator, n_keywords=100):
        self.evaluator = evaluator
        self.population = []
        self.n_keywords = n_keywords
    
    def initialize(self, samples):
        self.population += samples
        self.evaluator.initialize(samples)
        
    def exists(self, offspring):
        for s in self.population:
            if np.sum(s == offspring) == len(s):
                return True
        return False
        
    def add_offspring(self, offspring):
        if not self.exists(offspring):
            self.population.append(offspring)
            self.evaluator.add_candidate(offspring)
    
    def optimize(self, it, init=None):
        p = trange(it) if init is None else trange(init + 1, it)
        for i in p:
            parent1, parent2 = self.selection()
            offspring1, offspring2 = self.crossover(parent1, parent2)
            offspring1 = self.mutation(offspring1)
            offspring2 = self.mutation(offspring2)
            
            self.add_offspring(offspring1)
            self.add_offspring(offspring2)
            self.save_population()
            self.serialize(f'{i}.pkl')
            
    def selection(self):
        metric_vals = self.evaluator.values
        ind = np.argpartition(metric_vals, -2)[-2:]
        return self.population[ind[0]], self.population[ind[1]]
    
    def crossover(self, parent1, parent2):
        p1 = 0
        p2 = 0
        while p1 == p2:
            p1 = np.random.randint(0, len(parent1))
            p2 = np.random.randint(0, len(parent1))
        if p1 > p2:
            p1, p2 = p2, p1
        
        off1 = parent1.copy()
        off2 = parent2.copy()
        off1[p1:p2 + 1], off2[p1:p2 + 1] = off2[p1:p2 + 1], off1[p1:p2 + 1].copy()
        
        return off1, off2
    
    def mutation(self, offspring):
        mask = sps.bernoulli(0.01).rvs(size=len(offspring))
        for i, m in enumerate(mask):
            if m == 1:
                if not offspring[i]:
                    offspring[i] = True
                else:
                    offspring[i] = False
        return offspring
    
    def serialize(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self, f)
    
    def save_population(self):
        population_df = pd.DataFrame(self.population)
        population_df['score'] = self.evaluator.values
        population_df.to_csv('population.csv', index=None)
            
    @classmethod
    def deserialize(cls, filename):
        with open(filename, 'rb') as f:
            obj = pickle.load(f)
            return obj


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--toloka-token', type=str, required=True)
    parser.add_argument('--aws-access-key-id', type=str, required=True)
    parser.add_argument('--aws-secret-access-key', type=str, required=True)
    parser.add_argument('--endpoint-url', type=str, required=True)
    parser.add_argument('--bucket', type=str, required=True)
    parser.add_argument('--base-pool-id', type=str, required=True)
    args = parser.parse_args()

    api = DiffusionApi()
    generator = ImageGenerator(api)
    prompts_df = pd.read_csv('prompts.csv')
    img_saver = ImageSaver(args.aws_access_key_id, args.aws_secret_access_key, args.endpoint_url, args.bucket)
    hp_tasks = pd.read_csv('comp_hp.csv')

    toloka_client = toloka.TolokaClient(args.toloka_token, 'PRODUCTION') 
    task_processor = TaskProcessor(toloka_client, parser.base_pool_id, prompts_df['Filtered'], hp_tasks)
    keywords = list(pd.read_csv('keywords.csv')['keyword'])
    evaluator = Evaluator(prompts_df['Filtered'], prompts_df['Orientation'], keywords, generator, img_saver, task_processor)
    opt = GeneticOpt(evaluator, 100)

    opt.optimize(2000)
