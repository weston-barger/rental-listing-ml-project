
import re

import pandas     as pd
import numpy      as np

import os.path    as osp

import scipy      as sp
import scipy.sparse as sps


from sklearn.feature_extraction.text    import CountVectorizer
from sklearn.feature_extraction.text    import HashingVectorizer
from sklearn.feature_extraction.text    import TfidfTransformer
from sklearn.feature_extraction         import FeatureHasher
from sklearn.cluster                    import Birch
from sklearn.metrics                    import log_loss

from math import ceil


# a function for concatenating sparse matrices
def concatenate_csc(matrix1, matrix2):
    matrix1 = sps.csc_matrix(matrix1)
    matrix2 = sps.csc_matrix(matrix2)
    new_data = np.concatenate((matrix1.data, matrix2.data))
    new_indices = np.concatenate((matrix1.indices, matrix2.indices))
    new_ind_ptr = matrix2.indptr + len(matrix1.data)
    new_ind_ptr = new_ind_ptr[1:]
    new_ind_ptr = np.concatenate((matrix1.indptr, new_ind_ptr))
    return sps.csc_matrix((new_data, new_indices, new_ind_ptr))

class ProjectData():
    """
    class: ProjectData

    Used to manipulate and process the data corresponding to the kaggle
    competition ( https://www.kaggle.com/c/two-sigma-connect-rental-listing-inquiries )

    methods: __init__
            
            reset: a method for returning all instance variables to the default state

            add_text_features: build features corresponding to the text data

            add_handcrafted_features: build our hand-crafted features
            
            read_in_data: reads the data provided by the Kaggle competition

            build_matrices: create numpy/scipy matrices corresponding to Pandas dataframe

            sparse_prediction_proba: predicts class probabilities when the data matrices are sparse
                                     (sklearn requires that the matrices be numpy i.e. dense)

            sparse_prediction_scalar: predicts output classes when the data matrices are spares
                                      (sklearn requires that the matrices be numpy i.e. dense)

            kaggle_output_from_scalars: outputs predictions in Kaggle submission form from scalar predictions
            
            kaggle_output_from_probs: outputs predictions in Kaggle submission form from class probability predictions

            get_log_loss: compute log loss

            shuffle_matrices: shuffle the rows of the data matrices (and their corresponding labels)

            make_training_matrix_dense: convert the training matrix to a dense (numpy) matrix 
            
            make_test_matrix_dense: convert the test matrix to a dense (numpy) matrix 

            make_training_matrix_sparse: convert the training matrix to a sparse (scipy) matrix

            make_test_matrix_sparse: convert the test matrix to a spares (scipy) matrix 


    """
    def __init__(self, train_data, test_data,holdout_set = False,holdout_set_ratio = 0.2):
        """
        function : __init__
        arguments:  train_data (str) - path to training data json file

                    test_data  (str) - path to test data json file

                    holdout_set (bool, default = False): When true, the 
                    test set will be a holdout_set from the 
                    training set.

                    holdout_set_ratio (float, default = 0.2): Only matters 
                    when "holdout_set = True". The test set
                    will be of size 0.2 * "size of training set".
        """

        ###########################################
        # Error checking the inputs
        ###########################################
                
        # make sure input files exist
        if not osp.isfile(train_data):
            raise Exception('\n\nError: training file ' + str(train_data) + ' not found.')
        if not osp.isfile(test_data):
            raise Exception('\n\nError: training file ' + str(test_data) + ' not found.')

        ###########################################
        #
        # Definition: self.train_path,  self.test_path  
        # Type      : str, str 
        #
        #   Paths of training and test data files, respectively.
        #
        ############################################
        self.train_path = train_data
        self.test_path  = test_data
        
        # definitions above...
        self.holdout_set       = holdout_set
        self.holdout_set_ratio = holdout_set_ratio

        self.reset()

    def reset(self,):
        """
        function: reset
        arguments:  self
                    
        returns: None

        Performs the duties of __init__ i.e. setting the default
        instance var values, reading in the json data and
        creating test labels. This function is called by 
        __init__()
        """

       

        ###########################################
        #
        # Definition: self.dense_matrix_columns
        # Type      : list 
        #
        #   A list of strings corresponding to the 
        #   columns in the data to include in the 
        #   dense part of the matrices. These column
        #   names may appear in the original dataset
        #   or may correspond to hand-crafted features.
        ############################################
        self.dense_matrix_columns = [   'bathrooms',
                                        'bedrooms',
                                        'total_rooms',
                                        'price',
                                        'price_per_room',
                                        'median_neighborhood_price',
                                        'relative_neighborhood_price',
                                        'num_photos',
                                        'num_features',
                                        'num_description_words',
                                        'neighborhood_cluster',
                                        'year',
                                        'month',
                                        'day',
                                        'weekday',
                                        'hour',
                                        'street_address_count',
                                        'display_address_count',
                                        'building_id_count',
                                        'manager_id_count',
                                        'bedrooms_count',
                                        'bathrooms_count',
                                        'email',
                                        'phone',
                                        'redacted'
                                    ]


        ###########################################
        #
        # Definition: self.train_desc_counts, self.test_desc_counts
        # Type      : scipy sparse matrix 
        #
        #   These variables contain sparse matrix
        #   representations of the description features.
        #
        ############################################
        self.train_desc_counts = None
        self.test_desc_counts = None

        ###########################################
        #
        # Definition: self.log_price
        # Type      : bool 
        #
        #   When True, the price data will log transformed
        #
        ############################################
        self.log_price = True


        ###########################################
        #
        # Definition: self.tfidf
        # Type      : bool 
        #
        #   When true, we do tfidf analysis on "description"
        #   text data
        #
        ############################################
        self.tfidf = False

        ###########################################
        #
        # Definition: self.feature_hash_n
        # Type      : int 
        #
        #   The length of the hashing trick hash
        #
        ############################################
        self.feature_hash_n = -1


        ###########################################
        #
        # Definition: self.num_neighborhood_clusters
        # Type      : int 
        #
        #   The number of neighborhood clusters to put in the 
        #   data.
        #
        ############################################ 
        self.num_neighborhood_clusters = 100

        ###########################################
        #
        # Definition: self.interest_to_int, self.int_to_interest, self.int_to_vec
        # Type      : dict, dict, dict 
        #
        #   self.interest_to_int: maps interest level strings to integers
        #   self.int_to_interest: maps integers to interest level strings
        #   self.int_to_vec     : maps integers to list (for kaggle output)
        ############################################
        self.interest_to_int = {'high': 2, 'medium': 1, 'low': 0}
        self.int_to_interest = {2: 'high', 1: 'medium', 0: 'low'}
        self.int_to_vec      = {2:[1.,0.,0.], 1: [0.,1.,0.], 0: [0.,0.,1.]}


        ###########################################
        #
        # Definition: self.train_labels
        # Type      : ndarray 
        #
        #   contains the labels for the training set 
        #   associated with each row
        #
        #   high   - 2
        #   medium - 1
        #   low    - 0
        ############################################
        self.train_labels = None

        ###########################################
        #
        # Definition: self.test_labesl
        # Type      : ndarray 
        #
        #    contains labels associated with the
        #   test set (only if holdout_set = True)
        #
        ############################################
        self.test_labels = None

        ###########################################
        #
        # Definition: self.train_data   
        # Type      : DataFrame 
        #
        #   A pandas DataFrame containing the training data
        #
        ############################################
        self.train_data = None 
        self.train_n    = -1 

        ###########################################
        #
        # Definition: self.test_data
        # Type      : DataFrame
        #
        #   A pandas DataFrame containing the test data
        #
        ############################################
        self.test_data = None
        self.test_n    = -1

        ###########################################
        #
        # Definition: self.train_matrix
        # Type      : ndarray 
        #
        #   The numpy array corresponding to the training data
        #
        ############################################
        self.train_matrix = None

        ###########################################
        #
        # Definition: self.test_matrix
        # Type      : ndarray
        #
        #   The numpy array corresponding to the test data
        #
        ############################################
        self.test_matrix = None


        ###########################################
        # Perform the data input and processing
        ###########################################

        # read the training and test data from the json files
        # and create the label vectors
        self.read_in_data()
        

    def add_text_features(self):
        """
        function: add_text_features

        arguments: self
        returns: None 

        This function processes the "description" feature of the data. It stores the 
        output in spares matrix format in the variables self.train_desc_counts and self.test_desc_counts.
        If self.tfidf is True, then we perform tfidf anlysis. If self.feature_hash_n > -1, then 
        we hash the text down.

        """
        # perform a concatenation of the datasets
        total_data = pd.DataFrame(self.train_data.append(self.test_data))

        # compile a regex to look for html tags to remove
        htmlregex = re.compile(r'<.*?>')
        
        # get a list of the descriptions of the apartments
        # in which all HTML tags have been removed
        descriptions = [re.sub(htmlregex,' ',text) for text in list(total_data['description'])]
        del total_data

        # map put the descriptions in a One hot encoding
        cv = CountVectorizer()
        desc_counts = cv.fit(descriptions).transform(descriptions)
        print "Num unique words: " + str(len(cv.vocabulary_ ))

        if self.tfidf and ( not self.feature_hash_n == -1):
            raise Exception('\n\nError: cannot perform both tfidf and feature hashing.')
        
        # perform tf-idf analysis
        if self.tfidf:
            desc_counts = TfidfTransformer().fit_transform(desc_counts)

        # perform text hashing
        if not self.feature_hash_n == -1:
            ds = [dict(zip(str(desc_counts.getrow(i).indices), desc_counts.getrow(i).data)) for i in xrange(desc_counts.shape[0])]
            desc_counts = FeatureHasher(n_features  = self.feature_hash_n).transform(ds)

        self.train_desc_counts = desc_counts[:self.train_n,:]
        self.test_desc_counts  = desc_counts[self.train_n:,:]


    def add_handcrafted_features(self):
        """
        function: add_handcrafted_features
        arguments: self
        returns: None

        This function adds our hand-crafted features to the 
        pandas DataFrame object corresponding to both the training and test
        data sets. 


        """
        # make sure we have data
        if self.train_data is None or self.test_data is None:
            raise Exception('\n\nError: add_handcrafted_features called before data was read in.')

        # define the regex to identify email addresses
        emailregex = re.compile(r'[^@\s]+@[^@\s]+\.[^@\s]+')

        # define the regex to identify phone numbers
        phoneregex = re.compile(r'(\d{3}[-\.\s]??\d{3}[-\.\s]??\d{4}|\(\d{3}\)\s*\d{3}[-\.\s]??\d{4}|\d{3}[-\.\s]??\d{4})')

        ###########################################
        # First, we need to do some counting
        ###########################################
        
        # perform a concatenation of the datasets
        total_data = pd.DataFrame(self.train_data.append(self.test_data))
        del self.train_data
        del self.test_data

        # do a little bit of counting
        # list of lists where the first element is the new
        # feature name, and the second element is the 
        # feature for which the new feature is a value
        # count mapping of.
        features_to_count = {   
                                'street_address'  : None,
                                'display_address' : None,
                                'building_id' : None,
                                'manager_id'  : None,
                                'street_address'  : None,
                                'bedrooms' : None,
                                'bathrooms': None,
                            }


        # do the counting
        for feature in features_to_count:
            features_to_count[feature] = total_data[feature].value_counts()

        n = float(self.train_n + self.test_n)


        ###########################################
        # Date
        ###########################################
        
        # convert the date to pandas datetime format
        # This makes it easier to call things like "what day of the week is this?"
        total_data.created  = pd.to_datetime(total_data.created)

        #make a column for year
        total_data['year'] = total_data.created.dt.year

        # make a column for month created
        total_data['month'] = total_data.created.dt.month

        # make a column for day of the month created
        total_data['day']   = total_data.created.dt.day

        # make a column for day of the week (Monday=0, Sunday=6) created
        total_data['weekday']   = total_data.created.dt.dayofweek

        # make a column for the hour created
        total_data['hour']  = total_data.created.dt.hour

        ###########################################
        # rooms and photos
        ###########################################
        
        # make a column for total rooms
        total_data['total_rooms'] = total_data['bedrooms'] + total_data['bathrooms']

        # get the number of photos
        total_data['num_photos'] = total_data['photos'].apply(len)

        ###########################################
        # 
        ###########################################
        
        # get the number of features
        total_data['num_features'] = total_data['features'].apply(len)

        # get the number of words in description
        total_data["num_description_words"] = total_data["description"].apply(lambda x: len(x.split(" ")))

        # go to log price
        if self.log_price:
            total_data.loc[:,'price'] = total_data['price'].apply(np.log)

        ###########################################
        # Ratios
        ###########################################
        
        bedrooms_to_med_price = dict(total_data.groupby('bedrooms')['price'].median())
        total_data['price_to_median_ratio'] = total_data['bedrooms'].apply(lambda row: bedrooms_to_med_price[row])
        total_data['price_to_median_ratio'] = total_data['price'] / total_data['price_to_median_ratio']

        # build price_per_room
        # the +1 is simply to keep from overflow
        total_data['price_per_room'] = total_data['price']    / (total_data['total_rooms'] + 1)
        total_data['price_per_bed']  = total_data['price']    / (total_data['bedrooms']  + 1)
        total_data['price_per_bath'] = total_data['price']    / (total_data['bathrooms'] + 1)
        total_data['bed_bath_ratio'] = total_data['bedrooms'] / (total_data['bathrooms'] + 1)

        # make value count mapping features
        for feature in features_to_count:
            total_data[feature + '_count'] = total_data[feature].apply(lambda row: features_to_count[feature][row])

        total_data.loc[:,'bathrooms_count'] = total_data['bathrooms_count'].apply(lambda row: row / n )
        total_data.loc[:,'bedrooms_count']  = total_data['bedrooms_count'].apply(lambda row: row / n )

        ###########################################
        # neighborhood clustering
        ###########################################
        
        # partition the data into city and other
        in_city =(    (total_data['latitude']  > 40.55) 
                    & (total_data['latitude']  < 40.9)
                    & (total_data['longitude'] > -74.05) 
                    & (total_data['longitude'] < - 73.8))

        # add the cluster assignment
        bir = Birch(branching_factor=100, n_clusters=self.num_neighborhood_clusters, threshold=0.005)
        total_data['n_cluster'] = 0 
        total_data.loc[in_city,'neighborhood_cluster']  = bir.fit_predict(total_data.loc[in_city,['latitude','longitude']].as_matrix())
        total_data.loc[~in_city,'neighborhood_cluster'] = -1


        # CODE MIGHT NOT BE THAT ROBUST

        # add median_neighborhood_price and
        # relative_neighborhood_price
        total_data['median_neighborhood_price'] = 0
        for k in xrange(-1,self.num_neighborhood_clusters):
            in_k = (total_data['neighborhood_cluster'] == k)
            total_data.loc[in_k,'median_neighborhood_price'] = np.median(total_data.loc[in_k,'price'])

        total_data['relative_neighborhood_price'] = total_data['price'] / total_data['median_neighborhood_price']
        

        ###########################################
        # look for email addresses and phone numbers
        ###########################################

        # we set this feature to 1 if a email/phone is phone and -1 otherwise
        total_data['email'] = total_data['description'].apply(lambda row: 1 if emailregex.search(row) else -1)
        total_data['phone'] = total_data['description'].apply(lambda row: 1 if phoneregex.search(row) else -1)

        # whether or not the website has redacted the description
        total_data['redacted'] = -1
        total_data.loc[total_data['description'].str.contains('website_redacted'),'redacted'] = 1

        # partition total_data into training and test sets
        self.train_data = total_data.iloc[:self.train_n]
        self.test_data  = total_data.iloc[self.train_n:]

        # free up mem
        del total_data


    def read_in_data(self):
        """
        function: read_in_data
        arguments: self
        returns: None

        Reads the training and test data from json files
        """

        interest_map  = lambda x: self.interest_to_int[x]
        if not self.holdout_set:
            self.train_data = pd.read_json(self.train_path)
            self.test_data  = pd.read_json(self.test_path)
            self.test_labels = None
        else:
            data = pd.read_json(self.train_path)
            k = int(ceil(self.holdout_set_ratio * float(len(data))))
            self.test_data  = data.iloc[:k]
            self.train_data = data.iloc[k:] 
            self.test_labels = np.array(self.test_data.interest_level.apply(interest_map))

        self.train_labels = np.array(self.train_data.interest_level.apply(interest_map))            
        
        self.train_n = len(self.train_data)
        self.test_n  = len(self.test_data)

        return None

    def build_matrices(self):
        """
        function: build_matrices
        arguments: self
        returns: None

        Builds the training and test matrices from the pandas representation
        """
        num_columns = len(self.dense_matrix_columns)

        self.train_matrix    = np.ndarray(shape=(self.train_n,num_columns),dtype=np.float64 )
        self.train_matrix[:] = self.train_data[self.dense_matrix_columns]

        self.test_matrix    = np.ndarray(shape=(self.test_n,num_columns),dtype=np.float64 )
        self.test_matrix[:] = self.test_data[self.dense_matrix_columns]

        # if we have performed description analysis,
        # then concat that and make the matrices sparse
        if not (self.train_desc_counts is None or self.test_desc_counts is None):
            self.train_matrix = concatenate_csc(self.train_matrix, self.train_desc_counts)
            self.test_matrix  = concatenate_csc(self.test_matrix,  self.test_desc_counts)
         
        return None

    def sparse_prediction_proba(self, predictor,batchsize = 1000, verbose = False):
        """
        function: sparse_prediction_proba
        arguments:  self
                    predictor - a function handle of the predictor function 
                    batchsize (default: 1000) - the number of rows of self.test_matrix to make dense at once
                    verbose (default: False) - if True, this function prints out progress after each batch

        returns: the probabilities of each class for each test data point i.e. a (self.test_n, 3) numpy array
        """
        preds = np.ndarray(shape=(self.test_n,3))

        k, n = 0, ceil(self.test_n / float(batchsize))

        if verbose:
            print 'Percent completed:'
        while k < self.test_n:
            k2 = min(k+batchsize,self.test_n)
            preds[k:k2,:] =  predictor( self.test_matrix[k:k2,:].toarray() )
            k = k2
            if verbose:
                print '   ' + str(100. * (float(k) / self.test_n)) + '%'
        return preds

    def sparse_prediction_scalar(self, predictor,batchsize = 1000, verbose = False):
        """
        function: sparse_prediction_scalar
        arguments:  self
                    predictor - a function handle of the predictor function 
                    batchsize (default: 1000) - the number of rows of self.test_matrix to make dense at once
                    verbose (default: False) - if True, this function prints out progress after each batch

        returns: the class prediction each test data point i.e. a (self.test_n, 3) numpy array
        """
        preds = np.ndarray(shape=(self.test_n,1))

        k, n = 0, ceil(self.test_n / float(batchsize))

        if verbose:
            print 'Percent completed:'
        while k < self.test_n:
            k2 = min(k+batchsize,self.test_n)
            preds[k:k2,:] =  predictor( self.test_matrix[k:k2,:].toarray() )
            k = k2
            if verbose:
                print '   ' + str(100. * (float(k) / self.test_n)) + '%'
        return preds

    def kaggle_output_from_scalars(self,in_labels,filepath):
        """
        function: predict_from_scalars
        arguments:  self
                    in_labels (list or 1d-np.array): the class predictions
                    filepath (str) : path of the output csv (for kaggle)
        returns: None

        This function takes in scalar class predictions (0,1 or 2) and 
        outputs a csv file ready for kaggle submission.
        """
        pred_dict = dict(zip(list(self.test_data.listing_id), map(lambda x: self.int_to_vec[x],in_labels)))
        df = pd.DataFrame.from_dict(pred_dict, orient='index')
        df.columns = ['high','medium','low']        
        df.to_csv(filepath,index_label='listing_id')

        return None

    def kaggle_output_from_probs(self,in_probs,in_columns,filepath):
        """
        function: predict_from_probs
        arguments:  self
                    in_probs (ndarray): an n x 3 array with the corresponding class in_probs
                    in_columns (1d-array): an array which has the class corresponding to each column
                                            
                                            e.g. if in_colmns = [0,1,2], then this implies 
                                            the first column corresponds to low interest, 1st column to
                                            med interest, and 3rd column to high interest. 
                                            
                                            Note: in sklearn, classifiers have an instance variable called
                                            classes_ that has this information e.g. 

                                            svm_clf = SGDClassifier(**svm_args).fit(projd.train_matrix,projd.train_labels)
                                            svm_clf.classes_ (= [0,1,2])
                    filepath (str): path of the output csv file (for kaggle)
        returns: None

        This function takes in class probability predictions (p_low, p_med, p_high for instance) and 
        outputs a csv file ready for kaggle submission.
        """
        colargs = np.argsort(in_columns)[::-1]
        pred_dict = dict(zip(list(self.test_data.listing_id), in_probs[:,colargs]))
        df = pd.DataFrame.from_dict(pred_dict, orient='index')
        df.columns = ['high','medium','low']        
        df.to_csv(filepath,index_label='listing_id')

        return None

    def get_log_loss(self,predictions,in_columns = None):
        """
        function: get_log_loss
        arguments:  self
                    predictions (ndarray): predictions made on the test set
                    in_columns (1d-array,default = None): 
                                            an array which has the class corresponding to each column
                                            
                                            e.g. if in_colmns = [0,1,2], then this implies 
                                            the first column corresponds to low interest, 1st column to
                                            med interest, and 3rd column to high interest. 
                                            
                                            Note: in sklearn, classifiers have an instance variable called
                                            classes_ that has this information e.g. 

                                            svm_clf = SGDClassifier(**svm_args).fit(projd.train_matrix,projd.train_labels)
                                            svm_clf.classes_ (= [0,1,2])
        returns None
        """
        if self.train_labels is None:
            raise Exception('\n\nCannot perform log-loss operation when training set is not a holdout set.')
        
        if in_columns is None:
            return log_loss(self.test_labels, predictions )
        else:
            return log_loss(self.test_labels, predictions,labels = in_columns)

        return None

    def shuffle_matrices(self, seed = None):
        """
        function: shuffle_matrices
        arguments:  self
                    seed (positive int, default = None): specifies the seed
        returns None

        This function shuffles the training and test matrix datasets.
        """
        if not self.holdout_set:
            raise Exception('\n\nError: shuffle_set can only be used when the test set is constructed from a holdout set.')

        if not sps.issparse(self.train_matrix) and not sps.issparse(self.test_matrix):
            sp = False
            matrix = np.ndarray(shape = (self.test_n + self.train_n, self.train_matrix.shape[1]), dtype = np.float64)
            
        else:
            sp = True
            matrix = sps.lil_matrix( (self.test_n + self.train_n, self.train_matrix.shape[1]) ,dtype = np.float64)
            self.train_matrix = sps.lil_matrix(self.train_matrix)
            self.test_matrix = sps.lil_matrix(self.test_matrix)

        labels = np.ndarray(shape = (self.test_n + self.train_n,),dtype = np.float64)

        matrix[:self.test_n,:] = self.test_matrix[:]
        matrix[self.test_n:,:] = self.train_matrix[:]

        labels[:self.test_n] = self.test_labels
        labels[self.test_n:] = self.train_labels

        if not seed is None:
            np.random.seed(seed)
        inds = np.random.permutation(matrix.shape[0])
        
        self.test_matrix[:]  = matrix[inds[:self.test_n],:]
        self.train_matrix[:] = matrix[inds[self.test_n:],:]

        if sp:
            self.train_matrix = sps.csc_matrix(self.train_matrix)
            self.test_matrix  = sps.csc_matrix(self.test_matrix)

        self.test_labels[:]  = labels[inds[:self.test_n]]
        self.train_labels[:] = labels[inds[self.test_n:]]

        return None



    def make_training_matrix_dense(self):
        """
        function: make_training_matrix_dense
        arguments: self
        returns: None

        Makes self.train_matrix a dense numpy array
        """
        if self.train_matrix is None:
            raise Exception('\n\nError: matrices need to exist to make them dense.')
        try:
            self.train_matrix = self.train_matrix.toarray()
        except MemoryError:
            print 'Error: not enough memory to make train_matrix dense. Keeping train_matirx sparse.'
            self.make_training_matrix_sparse()
        except AttributeError:
            print 'Warning: make_training_matrix_dense failed, matrix is already dense.'
        except:
            print 'Warning: make_training_matrix_dense failed. Keeping train_matrix sparse.'
            self.make_training_matrix_sparse()

        return None
    def make_test_matrix_dense(self):
        """
        function: make_test_matrix_dense
        arguments: self
        returns: None

        Makes self.test_matrix a dense numpy array
        """
        if self.test_matrix is None:
            raise Exception('\n\nError: matrices need to exist to make them dense.')
        try:
            self.test_matrix = self.test_matrix.toarray()
        except MemoryError:
            print 'Error: not enough memory to make test_matrix dense. Keeping test_matrix sparse.'
            self.make_test_matrix_sparse()
        except AttributeError:
            print 'Warning: make_test_matrix_dense failed, matrix is already dense.'
        except:
            print 'Warning: make_test_matrix_dense failed. Keeping test_matrix sparse.'
            self.make_test_matrix_sparse()
        return None
    def make_training_matrix_sparse(self):
        """
        function: make_training_matrix_sparse
        arguments: self
        returns: None

        Makes self.train_matrix a scipy csc matrix
        """
        if self.train_matrix is None:
            raise Exception('\n\nError: matrices need to exist to make them sparse.')
        self.train_matrix = sps.csc_matrix(self.train_matrix)

    def make_test_matrix_sparse(self):
        """
        function: make_test_matrix_sparse
        arguments: self
        returns: None

        Makes self.test_matrix a scipy csc matrix
        """
        if self.test_matrix is None:
            raise Exception('\n\nError: matrices need to exist to make them sparse.')
        self.test_matrix = sps.csc_matrix(self.test_matrix)

if __name__=='__main__':
    prd = ProjectData("train.json","test.json")

