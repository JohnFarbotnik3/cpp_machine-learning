
#include <filesystem>
#include "./utils/file_io.cpp"

namespace ML::config {
	using std::string;
	using std::filesystem::path;
	namespace fio = utils::file_io;

	struct ml_server_config {
		path dir_images;// location of training data.
		path dir_models;// where to save/load model checkpoints from.
		path dir_output;// where to save outputs.

		ml_server_config() {
			dir_images = "./data/images";
			dir_models = "./data/models";
			dir_output = "./data/output";
		}

		int load_config_file(string filepath) {
			int status;
			string contents = fio::read_file(filepath, status);
			if(status != 0) return status;
			// TODO: load config
			return 0;
		}
	};

}
