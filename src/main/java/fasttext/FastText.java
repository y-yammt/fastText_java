package fasttext;

import java.io.*;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import java.util.Set;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.atomic.AtomicLong;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

import fasttext.Args.model_name;
import fasttext.Dictionary.EntryType;
import fasttext.io.BufferedLineReader;
import fasttext.io.LineReader;
import ru.avicomp.io.FSReader;

/**
 * FastText class, can be used as a lib in other projects
 *
 * @author Ivan
 */
public class FastText {

    private Args args_;
    private Dictionary dict_;
    private Matrix input_;
    private QMatrix qinput_;
    private Matrix output_;
    private Model model_;

    private AtomicLong tokenCount_;
    private long start_;
    private boolean quant_;

    private Class<? extends LineReader> lineReaderClass_ = BufferedLineReader.class;
    private long threadFileSize;

    /**
     * <pre>{@code
     * void FastText::getVector(Vector& vec, const std::string& word) const {
     *  const std::vector<int32_t>& ngrams = dict_->getSubwords(word);
     *  vec.zero();
     *  for (auto it = ngrams.begin(); it != ngrams.end(); ++it) {
     *      if (quant_) {
     *          vec.addRow(*qinput_, *it);
     *      } else {
     *          vec.addRow(*input_, *it);
     *      }
     *  }
     *  if (ngrams.size() > 0) {
     *      vec.mul(1.0 / ngrams.size());
     *  }
     * }
     * }</pre>
     *
     * @param word
     * @return
     */
    public Vector getVector(String word) {
        Vector res = new Vector(args_.dim);
        List<Integer> ngrams = dict_.getSubwords(word);
        for (Integer it : ngrams) {
            if (quant_) {
                res.addRow(qinput_, it);
            } else {
                res.addRow(input_, it);
            }
        }
        if (ngrams.size() > 0) {
            res.mul(1.0f / ngrams.size());
        }
        return res;
    }

    /**
     * <pre>{@code
     * void FastText::saveVectors() {
     *  std::ofstream ofs(args_->output + ".vec");
     *  if (!ofs.is_open()) {
     *      std::cerr << "Error opening file for saving vectors." << std::endl;
     *      exit(EXIT_FAILURE);
     *  }
     *  ofs << dict_->nwords() << " " << args_->dim << std::endl;
     *  Vector vec(args_->dim);
     *  for (int32_t i = 0; i < dict_->nwords(); i++) {
     *      std::string word = dict_->getWord(i);
     *      getVector(vec, word);
     *      ofs << word << " " << vec << std::endl;
     *  }
     *  ofs.close();
     * }
     * }</pre>
     *
     * @throws IOException
     */
    public void saveVectors() throws IOException {
        if (Utils.isEmpty(args_.output)) {
            if (args_.verbose > 1) {
                System.out.println("output is empty, skip save vector file");
            }
            return;
        }
        // validate and prepare:
        Path file = Paths.get(args_.output + ".vec");
        args_.getIOStreams().prepare(file.toString());
        if (!args_.getIOStreams().canWrite(file.toString())) {
            throw new IOException("Can't write to " + file);
        }
        if (args_.verbose > 1) {
            System.out.println("Saving Vectors to " + file.toAbsolutePath());
        }
        try (Writer writer = args_.createWriter(file)) {
            writer.write(dict_.nwords() + " " + args_.dim + "\n");
            for (int i = 0; i < dict_.nwords(); i++) {
                String word = dict_.getWord(i);
                Vector vec = getVector(word);
                writer.write(word);
                for (int j = 0; j < vec.m_; j++) {
                    writer.write(" ");
                    writer.write(IOUtil.formatNumber(vec.data_[j]));
                }
                writer.write("\n");
            }
            writer.flush();
        }
    }

    public void saveModel() throws IOException {
        if (Utils.isEmpty(args_.output)) {
            if (args_.verbose > 1) {
                System.out.println("output is empty, skip save model file");
            }
            return;
        }

        File file = new File(args_.output + ".bin");
        if (file.exists()) {
            file.delete();
        }
        if (file.getParentFile() != null) {
            file.getParentFile().mkdirs();
        }
        if (args_.verbose > 1) {
            System.out.println("Saving model to " + file.getCanonicalPath().toString());
        }
        OutputStream ofs = new BufferedOutputStream(new FileOutputStream(file));
        try {
            args_.save(ofs);
            dict_.save(ofs);
            input_.save(ofs);
            output_.save(ofs);
        } finally {
            ofs.flush();
            ofs.close();
        }
    }

    public void saveOutput() {
        // TODO:
    }

    /**
     * Load binary model file.
     *
     * @param filename
     * @throws IOException
     */
    public void loadModel(String filename) throws IOException {
        DataInputStream dis = null;
        BufferedInputStream bis = null;
        try {
            File file = new File(filename);
            if (!(file.exists() && file.isFile() && file.canRead())) {
                throw new IOException("Model file cannot be opened for loading!");
            }
            bis = new BufferedInputStream(new FileInputStream(file));
            dis = new DataInputStream(bis);

            args_ = new Args();
            dict_ = new Dictionary(args_);
            input_ = new Matrix();
            output_ = new Matrix();

            args_.load(dis);
            dict_.load(dis);
            input_.load(dis);
            output_.load(dis);

            model_ = new Model(input_, output_, args_, 0);
            if (args_.model == model_name.sup) {
                model_.setTargetCounts(dict_.getCounts(EntryType.LABEL));
            } else {
                model_.setTargetCounts(dict_.getCounts(EntryType.WORD));
            }
        } finally {
            if (bis != null) {
                bis.close();
            }
            if (dis != null) {
                dis.close();
            }
        }
    }

    public void printInfo(float progress, float loss) {
        float t = (float) (System.currentTimeMillis() - start_) / 1000;
        float ws = (float) (tokenCount_.get()) / t;
        float wst = (float) (tokenCount_.get()) / t / args_.thread;
        float lr = (float) (args_.lr * (1.0f - progress));
        int eta = (int) (t / progress * (1 - progress));
        int etah = eta / 3600;
        int etam = (eta - etah * 3600) / 60;
        System.out.printf("\rProgress: %.1f%% words/sec: %d words/sec/thread: %d lr: %.6f loss: %.6f eta: %d h %d m",
                100 * progress, (int) ws, (int) wst, lr, loss, etah, etam);
    }

    /**
     * <pre>{@code
     * void FastText::supervised(Model& model, real lr, const std::vector<int32_t>& line, const std::vector<int32_t>& labels) {
     *  if (labels.size() == 0 || line.size() == 0) return;
     *  std::uniform_int_distribution<> uniform(0, labels.size() - 1);
     *  int32_t i = uniform(model.rng);
     *  model.update(line, labels[i], lr);
     * }
     * }</pre>
     *
     * @param model
     * @param lr
     * @param line
     * @param labels
     */
    public void supervised(Model model, float lr, final List<Integer> line, final List<Integer> labels) {
        if (labels.size() == 0 || line.size() == 0)
            return;
        //int i = Utils.randomInt(model.rng, 1, labels.size()) - 1;
        int i = Utils.nextInt(model.rng, 0, labels.size() - 1);
        model.update(line, labels.get(i), lr);
    }

    /**
     * <pre>{@code
     * void FastText::cbow(Model& model, real lr, const std::vector<int32_t>& line) {
     *  std::vector<int32_t> bow;
     *  std::uniform_int_distribution<> uniform(1, args_->ws);
     *  for (int32_t w = 0; w < line.size(); w++) {
     *      int32_t boundary = uniform(model.rng);
     *      bow.clear();
     *      for (int32_t c = -boundary; c <= boundary; c++) {
     *          if (c != 0 && w + c >= 0 && w + c < line.size()) {
     *              const std::vector<int32_t>& ngrams = dict_->getSubwords(line[w + c]);
     *              bow.insert(bow.end(), ngrams.cbegin(), ngrams.cend());
     *          }
     *      }
     *      model.update(bow, line[w], lr);
     *  }
     * }
     * }</pre>
     *
     * @param model
     * @param lr
     * @param line
     */
    public void cbow(Model model, float lr, final List<Integer> line) {
        for (int w = 0; w < line.size(); w++) {
            List<Integer> bow = new ArrayList<>();
            int boundary = Utils.nextInt(model.rng, 1, args_.ws);
            for (int c = -boundary; c <= boundary; c++) {
                if (c != 0 && w + c >= 0 && w + c < line.size()) {
                    List<Integer> ngrams = dict_.getSubwords(line.get(w + c));
                    bow.addAll(ngrams);
                }
            }
            model.update(bow, line.get(w), lr);
        }
    }

    /**
     * <pre>{@code
     * void FastText::skipgram(Model& model, real lr, const std::vector<int32_t>& line) {
     *  std::uniform_int_distribution<> uniform(1, args_->ws);
     *  for (int32_t w = 0; w < line.size(); w++) {
     *      int32_t boundary = uniform(model.rng);
     *      const std::vector<int32_t>& ngrams = dict_->getSubwords(line[w]);
     *      for (int32_t c = -boundary; c <= boundary; c++) {
     *          if (c != 0 && w + c >= 0 && w + c < line.size()) {
     *              model.update(ngrams, line[w + c], lr);
     *          }
     *      }
     *  }
     * }}</pre>
     *
     * @param model
     * @param lr
     * @param line
     */
    public void skipgram(Model model, float lr, final List<Integer> line) {
        for (int w = 0; w < line.size(); w++) {
            int boundary = Utils.nextInt(model.rng, 1, args_.ws);
            List<Integer> ngrams = dict_.getSubwords(line.get(w));
            for (int c = -boundary; c <= boundary; c++) {
                if (c != 0 && w + c >= 0 && w + c < line.size()) {
                    model.update(ngrams, line.get(w + c), lr);
                }
            }
        }
    }

    public void test(InputStream in, int k) throws IOException, Exception {
        int nexamples = 0, nlabels = 0;
        double precision = 0.0f;
        List<Integer> line = new ArrayList<Integer>();
        List<Integer> labels = new ArrayList<Integer>();

        LineReader lineReader = null;
        try {
            lineReader = lineReaderClass_.getConstructor(InputStream.class, String.class).newInstance(in, args_.charset.name());
            String[] lineTokens;
            while ((lineTokens = lineReader.readLineTokens()) != null) {
                if (lineTokens.length == 1 && "quit".equals(lineTokens[0])) {
                    break;
                }
                dict_.getLine(lineTokens, line, labels, model_.rng);
                dict_.addNgrams(line, args_.wordNgrams);
                if (labels.size() > 0 && line.size() > 0) {
                    List<Pair<Float, Integer>> modelPredictions = new ArrayList<Pair<Float, Integer>>();
                    model_.predict(line, k, modelPredictions);
                    for (Pair<Float, Integer> pair : modelPredictions) {
                        if (labels.contains(pair.getValue())) {
                            precision += 1.0f;
                        }
                    }
                    nexamples++;
                    nlabels += labels.size();
                    // } else {
                    // System.out.println("FAIL Test line: " + lineTokens +
                    // "labels: " + labels + " line: " + line);
                }
            }
        } finally {
            if (lineReader != null) {
                lineReader.close();
            }
        }

        System.out.printf("P@%d: %.3f%n", k, precision / (k * nexamples));
        System.out.printf("R@%d: %.3f%n", k, precision / nlabels);
        System.out.println("Number of examples: " + nexamples);
    }

    /**
     * Thread-safe predict api
     *
     * @param lineTokens
     * @param k
     * @return
     */
    public List<Pair<Float, String>> predict(String[] lineTokens, int k) {
        List<Integer> words = new ArrayList<Integer>();
        List<Integer> labels = new ArrayList<Integer>();
        dict_.getLine(lineTokens, words, labels, model_.rng);
        dict_.addNgrams(words, args_.wordNgrams);

        if (words.isEmpty()) {
            return null;
        }

        Vector hidden = new Vector(args_.dim);
        Vector output = new Vector(dict_.nlabels());
        List<Pair<Float, Integer>> modelPredictions = new ArrayList<Pair<Float, Integer>>(k + 1);

        model_.predict(words, k, modelPredictions, hidden, output);

        List<Pair<Float, String>> predictions = new ArrayList<Pair<Float, String>>(k);
        for (Pair<Float, Integer> pair : modelPredictions) {
            predictions.add(new Pair<Float, String>(pair.getKey(), dict_.getLabel(pair.getValue())));
        }
        return predictions;
    }

    public void predict(String[] lineTokens, int k, List<Pair<Float, String>> predictions) throws IOException {
        List<Integer> words = new ArrayList<Integer>();
        List<Integer> labels = new ArrayList<Integer>();
        dict_.getLine(lineTokens, words, labels, model_.rng);
        dict_.addNgrams(words, args_.wordNgrams);

        if (words.isEmpty()) {
            return;
        }
        List<Pair<Float, Integer>> modelPredictions = new ArrayList<Pair<Float, Integer>>(k + 1);
        model_.predict(words, k, modelPredictions);
        predictions.clear();
        for (Pair<Float, Integer> pair : modelPredictions) {
            predictions.add(new Pair<Float, String>(pair.getKey(), dict_.getLabel(pair.getValue())));
        }
    }

    /**
     * TODO:
     *
     * @param in
     * @param k
     * @param print_prob
     * @throws IOException
     * @throws Exception
     */
    public void predict(InputStream in, int k, boolean print_prob) throws IOException, Exception {
        List<Pair<Float, String>> predictions = new ArrayList<Pair<Float, String>>(k);

        LineReader lineReader = null;

        try {
            lineReader = lineReaderClass_.getConstructor(InputStream.class, String.class).newInstance(in, args_.charset.name());
            String[] lineTokens;
            while ((lineTokens = lineReader.readLineTokens()) != null) {
                if (lineTokens.length == 1 && "quit".equals(lineTokens[0])) {
                    break;
                }
                predictions.clear();
                predict(lineTokens, k, predictions);
                if (predictions.isEmpty()) {
                    System.out.println("n/a");
                    continue;
                }
                for (Pair<Float, String> pair : predictions) {
                    System.out.print(pair.getValue());
                    if (print_prob) {
                        System.out.printf(" %f", Math.exp(pair.getKey()));
                    }
                }
                System.out.println();
            }
        } finally {
            if (lineReader != null) {
                lineReader.close();
            }
        }
    }

    public void wordVectors() {
        LineReader lineReader = null;
        try {
            lineReader = lineReaderClass_.getConstructor(InputStream.class, String.class).newInstance(System.in,
                    args_.charset.name());
            String word;
            while (!Utils.isEmpty((word = lineReader.readLine()))) {
                Vector vec = getVector(word);
                System.out.println(word + " " + vec);
            }
        } catch (Exception e) {
            e.printStackTrace();
        } finally {
            if (lineReader != null) {
                try {
                    lineReader.close();
                } catch (IOException e) {
                    e.printStackTrace();
                }
            }
        }
    }

    public void textVectors() {
        List<Integer> line = new ArrayList<Integer>();
        List<Integer> labels = new ArrayList<Integer>();
        Vector vec = new Vector(args_.dim);
        LineReader lineReader = null;
        try {
            lineReader = lineReaderClass_.getConstructor(InputStream.class, String.class).newInstance(System.in, args_.charset.name());
            String[] lineTokens;
            while ((lineTokens = lineReader.readLineTokens()) != null) {
                if (lineTokens.length == 1 && "quit".equals(lineTokens[0])) {
                    break;
                }
                dict_.getLine(lineTokens, line, labels, model_.rng);
                dict_.addNgrams(line, args_.wordNgrams);
                vec.zero();
                for (Integer it : line) {
                    vec.addRow(input_, it);
                }
                if (!line.isEmpty()) {
                    vec.mul(1.0f / line.size());
                }
                System.out.println(vec);
            }
        } catch (Exception e) {
            e.printStackTrace();
        } finally {
            if (lineReader != null) {
                try {
                    lineReader.close();
                } catch (IOException e) {
                    e.printStackTrace();
                }
            }
        }
    }

    public void printVectors() {
        if (args_.model == model_name.sup) {
            textVectors();
        } else {
            wordVectors();
        }
    }

    public void loadVectors(String filename) throws IOException {
        List<String> words;
        Matrix mat; // temp. matrix for pretrained vectors
        int n, dim;

        BufferedReader dis = null;
        String line;
        String[] lineParts;
        try {
            dis = new BufferedReader(new InputStreamReader(new FileInputStream(filename), "UTF-8"));

            line = dis.readLine();
            lineParts = line.split(" ");
            n = Integer.parseInt(lineParts[0]);
            dim = Integer.parseInt(lineParts[1]);

            words = new ArrayList<String>(n);

            if (dim != args_.dim) {
                throw new IllegalArgumentException(
                        "Dimension of pretrained vectors does not match args -dim option, pretrain dim is " + dim
                                + ", args dim is " + args_.dim);
            }

            mat = new Matrix(n, dim);
            for (int i = 0; i < n; i++) {
                line = dis.readLine();
                lineParts = line.split(" ");
                String word = lineParts[0];
                for (int j = 1; j <= dim; j++) {
                    mat.data_[i][j - 1] = Float.parseFloat(lineParts[j]);
                }
                words.add(word);
                dict_.add(word);
            }

            dict_.threshold(1, 0);
            input_ = new Matrix(dict_.nwords() + args_.bucket, args_.dim);
            input_.uniform(1.0f / args_.dim);
            for (int i = 0; i < n; i++) {
                int idx = dict_.getId(words.get(i));
                if (idx < 0 || idx >= dict_.nwords())
                    continue;
                for (int j = 0; j < dim; j++) {
                    input_.data_[idx][j] = mat.data_[i][j];
                }
            }

        } catch (IOException e) {
            throw new IOException("Pretrained vectors file cannot be opened!", e);
        } finally {
            try {
                if (dis != null) {
                    dis.close();
                }
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
    }

    /**
     * @param args
     * @throws Exception
     */
    public void train(Args args) throws Exception {
        args_ = args;
        dict_ = new Dictionary(args_);

        if ("-".equals(args_.input)) {
            throw new IOException("Cannot use stdin for training!");
        }

        if (!args_.getIOStreams().canRead(args_.input)) {
            throw new IOException("Input file cannot be opened! " + args_.input);
        }

        dict_.readFromFile(args_);

        try (FSReader r = args_.createReader()) {
            threadFileSize = r.size();
        }

        if (!Utils.isEmpty(args_.pretrainedVectors)) {
            loadVectors(args_.pretrainedVectors);
        } else {
            input_ = new Matrix(dict_.nwords() + args_.bucket, args_.dim);
            input_.uniform(1.0f / args_.dim);
        }

        if (args_.model == model_name.sup) {
            output_ = new Matrix(dict_.nlabels(), args_.dim);
        } else {
            output_ = new Matrix(dict_.nwords(), args_.dim);
        }

        start_ = System.currentTimeMillis();
        tokenCount_ = new AtomicLong(0);
        if (args_.thread > 1) {
            ExecutorService service = //Executors.newFixedThreadPool(args_.thread);
                    Executors.newFixedThreadPool(args_.thread,
                            r -> {
                                Thread t = Executors.defaultThreadFactory().newThread(r);
                                t.setDaemon(true);
                                return t;
                            });
            Set<Future<Integer>> res = IntStream.range(0, args_.thread)
                    .mapToObj(id -> service.submit(() -> {
                        Thread.currentThread().setName("FT-TrainThread-" + id);
                        trainThread(id);
                        return id;
                    })).collect(Collectors.toSet());
            service.shutdown();
            for (Future<Integer> f : res) {
                try {
                    f.get();
                } catch (InterruptedException e) {
                    System.err.println("Interrupted");
                } catch (ExecutionException e) {
                    res.forEach(_f -> _f.cancel(true));
                    throw e;
                }
            }
        } else {
            trainThread(0);
        }
        model_ = new Model(input_, output_, args_, 0);

        if (args.verbose > 1) {
            long trainTime = (System.currentTimeMillis() - start_) / 1000;
            System.out.printf("\nTrain time used: %d sec\n", trainTime);
        }

        saveModel();
        saveVectors();
        if (args_.saveOutput > 0) {
            saveOutput();
        }
    }

    /**
     * The original code:
     * <pre>{@code
     * void FastText::trainThread(int32_t threadId) {
     *  std::ifstream ifs(args_->input);
     *  utils::seek(ifs, threadId * utils::size(ifs) / args_->thread);
     *  Model model(input_, output_, args_, threadId);
     *  if (args_->model == model_name::sup) {
     *      model.setTargetCounts(dict_->getCounts(entry_type::label));
     *  } else {
     *      model.setTargetCounts(dict_->getCounts(entry_type::word));
     *  }
     *  const int64_t ntokens = dict_->ntokens();
     *  int64_t localTokenCount = 0;
     *  std::vector<int32_t> line, labels;
     *  while (tokenCount < args_->epoch * ntokens) {
     *      real progress = real(tokenCount) / (args_->epoch * ntokens);
     *      real lr = args_->lr * (1.0 - progress);
     *      if (args_->model == model_name::sup) {
     *          localTokenCount += dict_->getLine(ifs, line, labels, model.rng);
     *          supervised(model, lr, line, labels);
     *      } else if (args_->model == model_name::cbow) {
     *          localTokenCount += dict_->getLine(ifs, line, model.rng);
     *          cbow(model, lr, line);
     *      } else if (args_->model == model_name::sg) {
     *          localTokenCount += dict_->getLine(ifs, line, model.rng);
     *          skipgram(model, lr, line);
     *      }
     *      if (localTokenCount > args_->lrUpdateRate) {
     *          tokenCount += localTokenCount;
     *          localTokenCount = 0;
     *          if (threadId == 0 && args_->verbose > 1) {
     *              printInfo(progress, model.getLoss());
     *          }
     *      }
     *  }
     *  if (threadId == 0 && args_->verbose > 0) {
     *      printInfo(1.0, model.getLoss());
     *      std::cerr << std::endl;
     *  }
     *  ifs.close();
     * }
     * }</pre>
     *
     * @param threadId
     * @throws IOException
     */
    protected void trainThread(int threadId) throws IOException {
        try (FSReader r = args_.createReader()) {
            long skip = threadId * threadFileSize / args_.thread;
            r.skipBytes(skip);
            Model model = new Model(input_, output_, args_, threadId);
            if (args_.model == model_name.sup) {
                model.setTargetCounts(dict_.getCounts(EntryType.LABEL));
            } else {
                model.setTargetCounts(dict_.getCounts(EntryType.WORD));
            }

            long ntokens = dict_.ntokens();
            long localTokenCount = 0;
            List<Integer> line = new ArrayList<>();
            List<Integer> labels = new ArrayList<>();
            while (tokenCount_.longValue() < args_.epoch * ntokens) {
                float progress = tokenCount_.floatValue() / (args_.epoch * ntokens);
                float lr = (float) (args_.lr * (1.0 - progress));
                if (args_.model == model_name.sup) {
                    localTokenCount += dict_.getLine(r, line, labels);
                    supervised(model, lr, line, labels);
                } else if (args_.model == model_name.cbow) {
                    localTokenCount += dict_.getLine(r, line, model.rng);
                    cbow(model, lr, line);
                } else if (args_.model == model_name.sg) {
                    localTokenCount += dict_.getLine(r, line, model.rng);
                    skipgram(model, lr, line);
                }
                if (localTokenCount > args_.lrUpdateRate) {
                    tokenCount_.addAndGet(localTokenCount);
                    localTokenCount = 0;
                    if (threadId == 0 && args_.verbose > 1) {
                        printInfo(progress, model.getLoss());
                    }
                }
            }
        }
    }

    public Args getArgs() {
        return args_;
    }

    public void setArgs(Args args) {
        this.args_ = args;
    }

    public Dictionary getDict() {
        return dict_;
    }

    public void setDict(Dictionary dict) {
        this.dict_ = dict;
    }

    public Matrix getInput() {
        return input_;
    }

    public void setInput(Matrix input) {
        this.input_ = input;
    }

    public Matrix getOutput() {
        return output_;
    }

    public void setOutput(Matrix output) {
        this.output_ = output;
    }

    public Model getModel() {
        return model_;
    }

    public void setModel(Model model) {
        this.model_ = model;
    }

}
