from enum import unique
import typing
import numpy as np
from langbrainscore.interface.encoder import _ANNEncoder
from collections import defaultdict
import string
import torch
from tqdm import tqdm
import xarray as xr
import getpass

class HuggingFaceEncoder(_ANNEncoder):
	_source_model = None
	
	def __init__(self, source_model: str = None,
				 sent_embed: str = 'last',
				 cache: str = None,
				 cache_new: bool = False) -> None:
		self._source_model = source_model
		
		# Pretrained model
		from transformers import AutoModel, AutoConfig, AutoTokenizer
		self.config = AutoConfig.from_pretrained(self._source_model)
		self.tokenizer = AutoTokenizer.from_pretrained(self._source_model)
		self.model = AutoModel.from_pretrained(self._source_model, config=self.config)
		self.sent_embed = sent_embed
		
		# Cache
		self.user = getpass.getuser()
		self.cache = cache
		self.cache_new = cache_new
	
	# assert (self.config.num_hidden_layers == len(hidden_states))
	
	def _aggregate_layers(self, hidden_states: dict, sent_embed: str = 'last-tok') -> None:
		"""[summary]
		Args:
			hidden_states (torch.Tensor): pytorch tensor of shape (n_items, dims)
			sent_embed: an object specifying the method to use for aggregating
													representations across items within a layer
		Raises:
			NotImplementedError
		Returns:
			np.ndarray: the aggregated array
		"""
		states_layers = dict()
		for i in hidden_states.keys():  # for each layer
			if sent_embed == 'last-tok':
				state = hidden_states[i][-1, :]  # get last token
			elif sent_embed == 'mean-tok':
				state = torch.mean(hidden_states[i], dim=0)  # mean over tokens
			elif sent_embed == 'median-tok':
				state = torch.median(hidden_states[i], dim=0)  # median over tokens
			elif sent_embed == 'sum-tok':
				state = torch.sum(hidden_states[i], dim=0)  # sum over tokens
			elif sent_embed == 'all-tok' or sent_embed == None:
				state = hidden_states
			else:
				raise NotImplementedError('Sentence embedding method not implemented')
			
			states_layers[i] = state.detach().numpy()
		
		return states_layers
	
	def _flatten_activations(self, states_sentences_agg: dict,
							 index: str = 'DEFAULTINDEX'):
		labels = []
		lst_arr_flat = []
		for layer, arr in states_sentences_agg.items():
			arr = np.array(arr)  # for each layer
			lst_arr_flat.append(arr)
			# Create multiindex for each layer. index 0 is the layer index, and index 1 is the unit index
			for i in range(arr.shape[0]):  # across units
				labels.append((layer, i))
		arr_flat = np.concatenate(lst_arr_flat)  # concatenated activations across layers
		df = pd.DataFrame(arr_flat).T
		df.index = [index]
		df.columns = pd.MultiIndex.from_tuples(labels)  # rows: stimuli, columns: units
		return df
	
	def _get_cache_path(self, make_cache: bool = True) -> str:
		if self.cache == 'auto':
			if self.user == 'gt':
				self.cache_path = f'{ROOTDIR}/model-actv/{self._source_model}/{self.sent_embed}'
			else:
				self.cache_path = f'/om2/user/{self.user}/model-actv/{self._source_model}/{self.sent_embed}'
			
			if make_cache:
				os.makedirs(self.cache_path, exist_ok=True)
		else:
			self.cache_path = self.cache  # supplied by user
		
		return self.cache_path
	
	def encode(self,
			   stimset: pd.DataFrame = None,
			   stim_col: str = 'sentence',
			   sent_embed: str = 'last-tok',
			   case: str = 'lower',
			   context_dim: str = None, bidirectional: bool = False):
		"""[summary]
		Args:
			sentence_level_args (ANNEmbeddingConfig, optional): [description]. Defaults to ANNEmbeddingConfig(aggregation='last').
			case (str, optional): [description]. Defaults to 'lower'.
			context_dimension (str, optional): the name of a dimension in our xarray-like dataset objcet that provides
												groupings of sampleids (stimuli) that should be used
												as context when generating encoder representations. for instance, in a word-by-word
												presentation paradigm we (may) still want the full sentence as context. [default: None].
			bidirectional (bool, optional): if True, allows using "future" context to generate the representation for a current token
											otherwise, only uses what occurs in the "past". some might say, setting this to False
											gives you a more biologically plausibly analysis downstream (: [default: False]
		Raises:
			NotImplementedError: [description]
			ValueError: [description]
		Returns:
			[type]: [description]
		"""
		stimsetid_all = [stimset.index[x].split('.')[0] for x in range(len(stimset))]
		assert (len(np.unique(stimsetid_all)) == 1)  # Check whether all sentences come from the same corpora
		self.stimset = stimset
		self.stimsetid = stimsetid_all[0]
		self.stim_col = stim_col
		
		actv_fname = f'{self.stimsetid}_actv.pkl'
		stim_fname = f'{self.stimsetid}_stim.pkl'
		
		if self.cache:
			self._get_cache_path()
			stim_fname = f'{self.cache_path}/{stim_fname}'
			actv_fname = f'{self.cache_path}/{actv_fname}'
			if os.path.exists(f'{actv_fname}'):
				print(f'Loading cached data for {self.stimsetid} from {self.cache_path}\n')
				stim = pd.read_pickle(stim_fname)
				actv = pd.read_pickle(actv_fname)
				assert (self.stimset.index == stim.index).all()
				assert (actv.index == stim.index).all()
				self.encoded_ann = actv
				return self.encoded_ann
		
		self.model.eval()
		
		stim_zipped = list(zip(self.stimset.index, self.stimset[self.stim_col].values))
		stim_unzipped = np.array([x[1] for x in stim_zipped])
		
		if context_dim is None:
			context_groups = np.arange(0, len(stim_zipped), 1)
		else:
			context_groups = self.stimset[context_dim].values
		
		states_sentences = defaultdict(list)
		###############################################################################
		# ALL SAMPLES LOOP
		###############################################################################
		states_sentences_across_groups = []
		stim_index_counter = 0
		for group in tqdm(np.unique(context_groups)):
			mask_context = context_groups == group
			stim_in_context = stim_unzipped[mask_context]  # mask based on the context group
			
			# we want to tokenize all stimuli of this ctxt group individually first in order to keep track of
			# which tokenized subunit belongs to what stimulus
			
			tokenized_lengths = [0]
			word_ids_by_stim = []  # contains a list of token->word_id lists per stimulus
			states_sentences_across_stim = []
			###############################################################################
			# CONTEXT LOOP
			###############################################################################
			for i, stimulus in enumerate(stim_in_context):
				if len(stim_in_context) > 1:
					print(f'encoding stimulus {i} of {len(stim_in_context)}')
				# mask based on the uni/bi-directional nature of models :)
				if not bidirectional:
					stim_directional = stim_in_context[:i + 1]
				else:
					stim_directional = stim_in_context
				
				stim_directional = ' '.join(stim_directional)
				
				if case == 'lower':
					stim_directional = stim_directional.lower()
				elif case == 'upper':
					stim_directional = stim_directional.upper()
				else:
					stim_directional = stim_directional
				
				# tokenize the stimuli
				tokenized_stim = self.tokenizer(stim_directional, padding=False, return_tensors='pt')
				tokenized_lengths += [tokenized_stim.input_ids.shape[1]]
				
				# Get the hidden states
				result_model = self.model(tokenized_stim.input_ids, output_hidden_states=True, return_dict=True)
				hidden_states = result_model[
					'hidden_states']  # dict with key=layer, value=3D tensor of dims: [batch, tokens, emb size]
				
				word_ids = tokenized_stim.word_ids()  # make sure this is positional, not based on word identity
				word_ids_by_stim += [
					word_ids]  # contains the ids for each tokenized words in stimulus todo: maybe get rid of this?
				
				# now cut the 'irrelevant' context from the hidden states
				for idx_layer, layer in enumerate(hidden_states):  # layer has dim [batch;tokens;emb dim]
					states_sentences[idx_layer] = layer[:, tokenized_lengths[-2]: tokenized_lengths[-1],
												  :].squeeze()  # to obtain [tokens;emb dim]
				
				# aggregate within a stimulus
				states_sentences_agg = self._aggregate_layers(states_sentences,
															  sent_embed=sent_embed)
				# dict with key=layer, value=array of # size [emb dim]
				
				# Convert to flattened pandas df
				current_stimid = stim_zipped[stim_index_counter][0]
				df_states_sentences_agg = self._flatten_activations(states_sentences_agg,
																	index=current_stimid)
				# df_states_sentences_agg.index = df_states_sentences_agg.index.map(
				
				# append the dfs to states_sentences_across_stim (which is ALL stim within a context group)
				states_sentences_across_stim.append(df_states_sentences_agg)
				# now we have all the hidden states for the current context group across all stimuli
				
				stim_index_counter += 1
			
			###############################################################################
			# END CONTEXT LOOP
			###############################################################################
			
			states_sentences_across_groups.append(pd.concat(states_sentences_across_stim, axis=0))
		
		###############################################################################
		# END ALL SAMPLES LOOP
		###############################################################################
		
		actv = pd.concat(states_sentences_across_groups, axis=0)
		
		print(f'Number of stimuli in activations: {actv.shape[0]}\n'
			  f'Number of units in activations: {actv.shape[1]}\n')
		
		# a = pd.read_pickle(
		# 	'/Users/gt/Documents/GitHub/control-neural/control_neural/model-actv-control/gpt2/last-tok/853_FED_20211013b_3T1_PL2017_activations.pkl')
		# assert (a.values == actv.values).all()
		
		assert (stimset.index == actv.index).all()
		
		if self.cache_new:
			stimset.to_pickle(stim_fname, protocol=4)
			actv.to_pickle(actv_fname, protocol=4)
			print(f'Cached activations to {actv_fname}')
		
		self.encoded_ann = actv
		
		return self.encoded_ann
