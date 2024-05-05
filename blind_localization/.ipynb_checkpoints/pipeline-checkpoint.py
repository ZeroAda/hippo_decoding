from blind_localization.preprocess import *
from blind_localization.model import *
from blind_localization.ProbablisticClassifier import *
from blind_localization.data_loading import *
from blind_localization.visualize import *
from blind_localization.alignment import *

def load_data_labels(source_session, config):
    source_file = load_session_data("hippo_decoding/data_path.json", source_session)
    public_file = load_session_data("hippo_decoding/data_path.json", "public")

    raw_signal, df, skipped_channels = load_data(source_file["raw_signal_path"], public_file["label_path"], source_file["xml_path"], sheet_name=source_file["sheet_name"])
    channel_region_map, skipped_channels, channel_channel_map = process_labels(df, public_file["mapping_path"], skipped_channels)
    raw_signal = process_signals(raw_signal, channel_channel_map)

    channel_map, D = load_channel_map(public_file["channel_map_path"])

    channel_ridx_map = get_channel_ridx_map(channel_region_map)
    channel_idx_train, channel_idx_val, channel_idx_test, _ = get_train_test_indices(channel_ridx_map, train_size=config["source_train_size"], custom_split=config["custom_split"])
    train_test_indices = channel_idx_train, channel_idx_val, channel_idx_test
    
    return raw_signal, D, channel_ridx_map, train_test_indices


def compute_neural_representation(raw_signal, source_session, config):
    source_file = load_session_data("hippo_decoding/data_path.json", source_session)
    corr_table = compute_similarity(raw_signal, np.arange(len(raw_signal)), source_file["model_path"], pre_train=config["pre_train"])
    print(corr_table.shape)
    # exit()
    channel_features = PCA(n_components=3).fit_transform(corr_table)
    # print(channel_features.shape)
    return channel_features

def alignment(channel_features, channel_ridx_map, X_train_target, y_train_target):
    X_source, y_source = channel_features[channel_ridx_map[:, 0]], channel_ridx_map[:, 1]
    tform = supervised_align_all(X_source, y_source, X_train_target, y_train_target)
    return tform


def model_evaluate(model, X_train, X_val, y_train_target, y_val_target, config):
    _, y_pred_train = model.predict(X_train[0], X_train[1], distance=config["distance"])
    _, y_pred_val = model.predict(X_val[0], X_val[1], distance=config["distance"])

    print("Training accuracy: ", accuracy_score(y_train_target, y_pred_train))
    print("Validation accuracy: ", accuracy_score(y_val_target, y_pred_val))
    return y_pred_train, y_pred_val


def run_pipeline_all(run_pipeline, session_names, config):
    y_target_ls, y_pred_ls, accuracy_ls = [], [], []
    
    for source_session in session_names:
        for target_session in session_names:
            print(source_session, target_session)
            y_target, y_pred = run_pipeline(source_session, target_session, config)

            y_target_ls.append(y_target)
            y_pred_ls.append(y_pred)
            accuracy_ls.append(accuracy_score(y_target, y_pred))

    return y_target_ls, y_pred_ls, accuracy_ls