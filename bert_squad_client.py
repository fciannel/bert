import tensorflow as tf
import tokenization
import collections
import grpc

from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc
from tensorflow.core.framework import tensor_pb2
from tensorflow.core.framework import tensor_shape_pb2
from tensorflow.core.framework import types_pb2


import run_squad as squad


channel = grpc.insecure_channel("172.17.0.9:8500")
stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)

def get_predictions(all_examples, all_features, all_results, n_best_size, max_answer_length, do_lower_case, null_score_diff_threshold):

    example_index_to_features = collections.defaultdict(list)
    for feature in all_features:
        example_index_to_features[feature.example_index].append(feature)

    unique_id_to_result = {}
    for result in all_results:
        unique_id_to_result[result.unique_id] = result

    _PrelimPrediction = collections.namedtuple(  # pylint: disable=invalid-name
        "PrelimPrediction",
        ["feature_index", "start_index", "end_index", "start_logit", "end_logit"])

    all_predictions = collections.OrderedDict()
    all_nbest_json = collections.OrderedDict()
    scores_diff_json = collections.OrderedDict()

    for (example_index, example) in enumerate(all_examples):
        features = example_index_to_features[example_index]

        prelim_predictions = []
        # keep track of the minimum score of null start+end of position 0
        score_null = 1000000  # large and positive
        min_null_feature_index = 0  # the paragraph slice with min mull score
        null_start_logit = 0  # the start logit at the slice with min null score
        null_end_logit = 0  # the end logit at the slice with min null score
        for (feature_index, feature) in enumerate(features):
            result = unique_id_to_result[feature.unique_id]
            start_indexes = squad._get_best_indexes(result.start_logits, n_best_size)
            end_indexes = squad._get_best_indexes(result.end_logits, n_best_size)
            # if we could have irrelevant answers, get the min score of irrelevant
            for start_index in start_indexes:
                for end_index in end_indexes:
                    # We could hypothetically create invalid predictions, e.g., predict
                    # that the start of the span is in the question. We throw out all
                    # invalid predictions.
                    if start_index >= len(feature.tokens):
                        continue
                    if end_index >= len(feature.tokens):
                        continue
                    if start_index not in feature.token_to_orig_map:
                        continue
                    if end_index not in feature.token_to_orig_map:
                        continue
                    if not feature.token_is_max_context.get(start_index, False):
                        continue
                    if end_index < start_index:
                        continue
                    length = end_index - start_index + 1
                    if length > max_answer_length:
                        continue
                    prelim_predictions.append(
                        _PrelimPrediction(
                            feature_index=feature_index,
                            start_index=start_index,
                            end_index=end_index,
                            start_logit=result.start_logits[start_index],
                            end_logit=result.end_logits[end_index]))

        prelim_predictions = sorted(
            prelim_predictions,
            key=lambda x: (x.start_logit + x.end_logit),
            reverse=True)

        _NbestPrediction = collections.namedtuple(  # pylint: disable=invalid-name
            "NbestPrediction", ["text", "start_logit", "end_logit"])

        seen_predictions = {}
        nbest = []
        for pred in prelim_predictions:
            if len(nbest) >= n_best_size:
                break
            feature = features[pred.feature_index]
            if pred.start_index > 0:  # this is a non-null prediction
                tok_tokens = feature.tokens[pred.start_index:(pred.end_index + 1)]
                orig_doc_start = feature.token_to_orig_map[pred.start_index]
                orig_doc_end = feature.token_to_orig_map[pred.end_index]
                orig_tokens = example.doc_tokens[orig_doc_start:(orig_doc_end + 1)]
                tok_text = " ".join(tok_tokens)

                # De-tokenize WordPieces that have been split off.
                tok_text = tok_text.replace(" ##", "")
                tok_text = tok_text.replace("##", "")

                # Clean whitespace
                tok_text = tok_text.strip()
                tok_text = " ".join(tok_text.split())
                orig_text = " ".join(orig_tokens)

                final_text = squad.get_final_text(tok_text, orig_text, do_lower_case)
                if final_text in seen_predictions:
                    continue

                seen_predictions[final_text] = True
            else:
                final_text = ""
                seen_predictions[final_text] = True

            nbest.append(
                _NbestPrediction(
                    text=final_text,
                    start_logit=pred.start_logit,
                    end_logit=pred.end_logit))

        # In very rare edge cases we could have no valid predictions. So we
        # just create a nonce prediction in this case to avoid failure.
        if not nbest:
            nbest.append(
                _NbestPrediction(text="empty", start_logit=0.0, end_logit=0.0))

        assert len(nbest) >= 1

        total_scores = []
        best_non_null_entry = None
        for entry in nbest:
            total_scores.append(entry.start_logit + entry.end_logit)
            if not best_non_null_entry:
                if entry.text:
                    best_non_null_entry = entry

        probs = squad._compute_softmax(total_scores)

        nbest_json = []
        for (i, entry) in enumerate(nbest):
            output = collections.OrderedDict()
            output["text"] = entry.text
            output["probability"] = probs[i]
            output["start_logit"] = entry.start_logit
            output["end_logit"] = entry.end_logit
            nbest_json.append(output)

        assert len(nbest_json) >= 1

        all_predictions[example.qas_id] = nbest_json[0]["text"]

        all_nbest_json[example.qas_id] = nbest_json

    return all_predictions, all_nbest_json


def read_squad_examples(input_data, is_training):
    """Read a SQuAD json file into a list of SquadExample."""
    # with tf.gfile.Open(input_file, "r") as reader:
    #     input_data = json.load(reader)["data"]

    def is_whitespace(c):
        if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
            return True
        return False

    examples = []
    for entry in input_data["data"]:
        for paragraph in entry["paragraphs"]:
            paragraph_text = paragraph["context"]
            doc_tokens = []
            char_to_word_offset = []
            prev_is_whitespace = True
            for c in paragraph_text:
                if is_whitespace(c):
                    prev_is_whitespace = True
                else:
                    if prev_is_whitespace:
                        doc_tokens.append(c)
                    else:
                        doc_tokens[-1] += c
                    prev_is_whitespace = False
                char_to_word_offset.append(len(doc_tokens) - 1)

            for qa in paragraph["qas"]:
                qas_id = qa["id"]
                question_text = qa["question"]
                start_position = None
                end_position = None
                orig_answer_text = None
                is_impossible = False
                if is_training:
                    if (len(qa["answers"]) != 1) and (not is_impossible):
                        raise ValueError(
                            "For training, each question should have exactly 1 answer.")
                    if not is_impossible:
                        answer = qa["answers"][0]
                        orig_answer_text = answer["text"]
                        answer_offset = answer["answer_start"]
                        answer_length = len(orig_answer_text)
                        start_position = char_to_word_offset[answer_offset]
                        end_position = char_to_word_offset[answer_offset + answer_length -
                                                           1]
                        # Only add answers where the text can be exactly recovered from the
                        # document. If this CAN'T happen it's likely due to weird Unicode
                        # stuff so we will just skip the example.
                        #
                        # Note that this means for training mode, every example is NOT
                        # guaranteed to be preserved.
                        actual_text = " ".join(
                            doc_tokens[start_position:(end_position + 1)])
                        cleaned_answer_text = " ".join(
                            tokenization.whitespace_tokenize(orig_answer_text))
                        if actual_text.find(cleaned_answer_text) == -1:
                            tf.logging.warning("Could not find answer: '%s' vs. '%s'",
                                               actual_text, cleaned_answer_text)
                            continue
                    else:
                        start_position = -1
                        end_position = -1
                        orig_answer_text = ""

                example = squad.SquadExample(
                    qas_id=qas_id,
                    question_text=question_text,
                    doc_tokens=doc_tokens,
                    orig_answer_text=orig_answer_text,
                    start_position=start_position,
                    end_position=end_position,
                    is_impossible=is_impossible)
                examples.append(example)

    return examples

def process_feature(feature):
    """Write a InputFeature to the TFRecordWriter as a tf.train.Example."""

    def create_int_feature(values):
        feature = tf.train.Feature(
            int64_list=tf.train.Int64List(value=list(values)))
        return feature

    features = collections.OrderedDict()
    features["unique_ids"] = create_int_feature([feature.unique_id])
    features["input_ids"] = create_int_feature(feature.input_ids)
    features["input_mask"] = create_int_feature(feature.input_mask)
    features["segment_ids"] = create_int_feature(feature.segment_ids)

    tf_example = tf.train.Example(features=tf.train.Features(feature=features))
    return tf_example.SerializeToString()


def run_single_prediction(input_data):
    RawResult = collections.namedtuple("RawResult", ["unique_id", "start_logits", "end_logits"])


    #input_data = {"data": [ {"title": "bla", "paragraphs": [{"context": "Up to 48 ports of full Power over Ethernet Plus (PoE+) capability. Resiliency with Field-Replaceable Units (FRU) and redundant power supply, fans, and modular uplinks. Flexible downlink options with data or PoE+. Operational efficiency with optional backplane stacking, supporting stacking bandwidth up to 160 Gbps. UADP 2.0 Mini with integrated CPU offers customers optimized scale with better cost structure. Enhanced security with AES-128 MACsec encryption, policy-based segmentation, and trustworthy systems. Layer 3 capabilities, including OSPF, EIGRP, ISIS, RIP, and routed access. Advanced network monitoring using Full Flexible NetFlow. Cisco Software-Defined Access (SD-Access): Simplified operations and deployment with policy-based automation from edge to cloud managed with Cisco Identity Services Engine (ISE). Network assurance and improved resolution time through Cisco DNA Center. Plug and Play (PnP) enabled: A simple, secure, unified, and integrated offering to ease new branch or campus device rollouts or updates to an existing network. Cisco IOS XE: A Common Licensing based operating system for the enterprise Cisco Catalyst 9000 product family with support for model-driven programmability and streaming telemetry. ASIC with programmable pipeline and micro-engine capabilities, along with template-based, configurable allocation of Layer 2 and Layer 3 forwarding, Access Control Lists (ACLs), and Quality of Service (QoS) entries.", "qas": [{"answers": [], "question": "How many ports does cisco 9200 switches support?", "id": "56bec9e83aeaaa14008c945f"}, {"answers": [], "question": "What are the FRU components that Catalyst 9200 Series Switches have?", "id": "56bec9e83aeaaa14008c9451"}, {"answers": [], "question": "Does Catalyst 9200 Series Switches have downlink options?", "id": "56bec9e83aeaaa14008c9452"}, {"answers": [], "question": "What is the stacking bandwidth for Catalyst 9200 Series Switches?", "id": "56bec9e83aeaaa14008c9453"}, {"answers": [], "question": "What does the Catalyst 9200 Series Switches offer?", "id": "56bec9e83aeaaa14008c9454"}, {"answers": [], "question": "List out the security features for Catalyst 9200 Series Switches?", "id": "56bec9e83aeaaa14008c9455"}, {"answers": [], "question": "under which routing protocol does the Catalyst 9200 Series Switches run?", "id": "56bec9e83aeaaa14008c9456"}, {"answers": [], "question": "What security monitoring used in Catalyst 9200 Series Switches?", "id": "56bec9e83aeaaa14008c9457"}, {"answers": [], "question": "What are some of the capabilities of Software-Defined Access?", "id": "56bec9e83aeaaa14008c9458"}, {"answers": [], "question": "Does Catalyst 9200 Series Switches have PnP?", "id": "56bec9e83aeaaa14008c9459"}, {"answers": [], "question": "What software does Catalyst 9200 Series Switches run?", "id": "56bec9e83aeaaa14008c945a"}, {"answers": [], "question": "What are the  capabilities of Catalyst 9200 Series Switches?", "id": "56bec9e83aeaaa14008c945b"}]}]}], "version":"1.1"}


    eval_examples = read_squad_examples(input_data, is_training=False)

    max_seq_length = 384
    max_query_length = 64
    doc_stride = 128
    vocab_file = 'gs://fciannel_storage/bert_models/uncased_L-24_H-1024_A-16/vocab.txt'
    do_lower_case = True
    max_answer_length = 30
    n_best_size = 20
    null_score_diff_threshold = 0.0

    tokenizer = tokenization.FullTokenizer(vocab_file=vocab_file, do_lower_case=do_lower_case)

    eval_features = []
    eval_tf_records = []

    def append_feature(feature):
      eval_features.append(feature)
      eval_tf_records.append(process_feature(feature))

    squad.convert_examples_to_features(
        examples=eval_examples,
        tokenizer=tokenizer,
        max_seq_length=max_seq_length,
        doc_stride=doc_stride,
        max_query_length=max_query_length,
        is_training=False,
        output_fn=append_feature)


    model_input = eval_tf_records[0]
    # model_input = eval_tf_records

    model_request = predict_pb2.PredictRequest()
    model_request.model_spec.name = 'squad'
    model_request.model_spec.signature_name = 'serving_default'
    dims = [tensor_shape_pb2.TensorShapeProto.Dim(size=1)]
    tensor_shape_proto = tensor_shape_pb2.TensorShapeProto(dim=dims)
    tensor_proto = tensor_pb2.TensorProto(
    dtype=types_pb2.DT_STRING,
    tensor_shape=tensor_shape_proto,
    string_val=[model_input])

    model_request.inputs['examples'].CopyFrom(tensor_proto)
    result = stub.Predict(model_request, 10.0)  # 10 secs timeout
    result_start = tf.make_ndarray(result.outputs["start_logits"])
    result_end = tf.make_ndarray(result.outputs["end_logits"])
    result_unique_ids = tf.make_ndarray(result.outputs["unique_ids"])

    start_logits = result_start.tolist()[0]
    end_logits = result_end.tolist()[0]
    unique_id = result_unique_ids.tolist()[0]

    all_results = []

    all_results.append(
        RawResult(
            unique_id=unique_id,
            start_logits=start_logits,
            end_logits=end_logits))

    all_predictions, all_nbest_json = get_predictions(eval_examples, eval_features, all_results, n_best_size, max_answer_length, do_lower_case, null_score_diff_threshold)
    return all_predictions, all_nbest_json



def main():
    input_data = {"data": [{"title": "bla", "paragraphs": [{"context": "There would be no more scoring in the third quarter, but early in the fourth, the Broncos drove to the Panthers 41-yard line. On the next play, Ealy knocked the ball out of Manning's hand as he was winding up for a pass, and then recovered it for Carolina on the 50-yard line. A 16-yard reception by Devin Funchess and a 12-yard run by Stewart then set up Gano's 39-yard field goal, cutting the Panthers deficit to one score at 16\u201310. The next three drives of the game would end in punts.", "qas": [{"answers": [{"text": "Ealy", "answer_start": 144}, {"text": "Ealy", "answer_start": 144}, {"text": "Ealy", "answer_start": 144}], "id": "56bec9e83aeaaa14008c945f", "question": "Who recovered a Manning fumble?"}]}]}], "version": "1.1"}
    all_predictions, all_nbest_json = run_single_prediction(input_data)
    print(all_predictions)

if __name__ == '__main__':
    main()
