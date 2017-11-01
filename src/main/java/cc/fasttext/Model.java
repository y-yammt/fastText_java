package cc.fasttext;

import java.util.*;

import cc.fasttext.Args.LossName;
import cc.fasttext.Args.ModelName;

public strictfp class Model {


    static final int SIGMOID_TABLE_SIZE = 512;
    static final int MAX_SIGMOID = 8;
    static final int LOG_TABLE_SIZE = 512;
    static final int NEGATIVE_TABLE_SIZE = 10_000_000;
    public static final Comparator<Pair<Float, Integer>> COMPARE_PAIRS = (l, r) -> r.first().compareTo(l.first());
    public static final Comparator<Float> HEAP_COMPARATOR = Comparator.reverseOrder();
    // todo: new
    public boolean quant_;
    public QMatrix qwi_;
    public QMatrix qwo_;
    public Random rng;
    private Matrix wi_; // input
    private Matrix wo_; // output
    private Args args_;
    private Vector hidden_;
    private Vector output_;
    private Vector grad_;
    private int hsz_; // dim
    @SuppressWarnings("unused")
    private int isz_; // input vocabSize
    private int osz_; // output vocabSize
    private float loss_;
    private long nexamples_;
    private float[] t_sigmoid;
    private float[] t_log;
    // used for negative sampling:
    private List<Integer> negatives;
    private int negpos;
    // used for hierarchical softmax:
    private List<List<Integer>> paths;
    private List<List<Boolean>> codes;
    private List<Node> tree;

    public Model(Matrix wi, Matrix wo, Args args, int seed) {
        hidden_ = new Vector(args.dim);
        output_ = new Vector(wo.m_);
        grad_ = new Vector(args.dim);

        rng = new Random(seed);

        wi_ = wi;
        wo_ = wo;
        args_ = args;
        isz_ = wi.m_;
        osz_ = wo.m_;
        hsz_ = args.dim;
        negpos = 0;
        loss_ = 0.0f;
        nexamples_ = 1L;
        initSigmoid();
        initLog();
    }

    /**
     * <pre>{@code
     * void Model::setQuantizePointer(std::shared_ptr<QMatrix> qwi, std::shared_ptr<QMatrix> qwo, bool qout) {
     *  qwi_ = qwi;
     *  qwo_ = qwo;
     *  if (qout) {
     *      osz_ = qwo_->getM();
     *  }
     * }}</pre>
     *
     * @param qwi
     * @param qwo
     * @param qout
     */
    public void setQuantizePointer(QMatrix qwi, QMatrix qwo, boolean qout) {
        this.qwi_ = qwi;
        this.qwo_ = qwo;
        if (qout) {
            osz_ = qwo_.getM();
        }
    }

    public float binaryLogistic(int target, boolean label, float lr) {
        float score = sigmoid(wo_.dotRow(hidden_, target));
        float alpha = lr * ((label ? 1.0f : 0.0f) - score);
        grad_.addRow(wo_, target, alpha);
        wo_.addRow(hidden_, target, alpha);
        if (label) {
            return -log(score);
        } else {
            return -log(1.0f - score);
        }
    }

    /**
     * <pre>{@code real Model::negativeSampling(int32_t target, real lr) {
     *  real loss = 0.0;
     *  grad_.zero();
     *  for (int32_t n = 0; n <= args_->neg; n++) {
     *      if (n == 0) {
     *          loss += binaryLogistic(target, true, lr);
     *      } else {
     *          loss += binaryLogistic(getNegative(target), false, lr);
     *      }
     *  }
     *  return loss;
     * }}</pre>
     *
     * @param target
     * @param lr
     * @return
     */
    public float negativeSampling(int target, float lr) {
        float loss = 0.0f;
        grad_.zero();
        for (int n = 0; n <= args_.neg; n++) {
            if (n == 0) {
                loss += binaryLogistic(target, true, lr);
            } else {
                loss += binaryLogistic(getNegative(target), false, lr);
            }
        }
        return loss;
    }

    /**
     * <pre>{@code real Model::hierarchicalSoftmax(int32_t target, real lr) {
     *  real loss = 0.0;
     *  grad_.zero();
     *  const std::vector<bool>& binaryCode = codes[target];
     *  const std::vector<int32_t>& pathToRoot = paths[target];
     *  for (int32_t i = 0; i < pathToRoot.size(); i++) {
     *      loss += binaryLogistic(pathToRoot[i], binaryCode[i], lr);
     *  }
     *  return loss;
     * }}</pre>
     *
     * @param target
     * @param lr
     * @return
     */
    public float hierarchicalSoftmax(int target, float lr) {
        float loss = 0.0f;
        grad_.zero();
        List<Boolean> binaryCode = codes.get(target);
        List<Integer> pathToRoot = paths.get(target);
        for (int i = 0; i < pathToRoot.size(); i++) {
            loss += binaryLogistic(pathToRoot.get(i), binaryCode.get(i), lr);
        }
        return loss;
    }

    /**
     * <pre>{@code void Model::computeOutputSoftmax(Vector& hidden, Vector& output) const {
     *  if (quant_ && args_->qout) {
     *      output.mul(*qwo_, hidden);
     *  } else {
     *      output.mul(*wo_, hidden);
     *  }
     *  real max = output[0], z = 0.0;
     *  for (int32_t i = 0; i < osz_; i++) {
     *      max = std::max(output[i], max);
     *  }
     *  for (int32_t i = 0; i < osz_; i++) {
     *      output[i] = exp(output[i] - max);
     *      z += output[i];
     *  }
     *  for (int32_t i = 0; i < osz_; i++) {
     *      output[i] /= z;
     *  }
     * }}</pre>
     *
     * @param hidden
     * @param output
     */
    public void computeOutputSoftmax(Vector hidden, Vector output) {
        if (quant_ && args_.qout) {
            output.mul(qwo_, hidden);
        } else {
            output.mul(wo_, hidden);
        }
        float max = output.get(0), z = 0.0f;
        for (int i = 0; i < osz_; i++) {
            max = Math.max(output.get(i), max);
        }
        for (int i = 0; i < osz_; i++) {
            output.set(i, (float) Math.exp(output.get(i) - max));
            z += output.get(i);
        }
        for (int i = 0; i < osz_; i++) {
            output.set(i, output.get(i) / z);
        }
    }

    public void computeOutputSoftmax() {
        computeOutputSoftmax(hidden_, output_);
    }

    /**
     * <pre>{@code real Model::softmax(int32_t target, real lr) {
     *  grad_.zero();
     *  computeOutputSoftmax();
     *  for (int32_t i = 0; i < osz_; i++) {
     *      real label = (i == target) ? 1.0 : 0.0;
     *      real alpha = lr * (label - output_[i]);
     *      grad_.addRow(*wo_, i, alpha);
     *      wo_->addRow(hidden_, i, alpha);
     *  }
     *  return -log(output_[target]);
     * }}</pre>
     *
     * @param target
     * @param lr
     * @return
     */
    public float softmax(int target, float lr) {
        grad_.zero();
        computeOutputSoftmax();
        for (int i = 0; i < osz_; i++) {
            float label = (i == target) ? 1.0f : 0.0f;
            float alpha = lr * (label - output_.get(i));
            grad_.addRow(wo_, i, alpha);
            wo_.addRow(hidden_, i, alpha);
        }
        return -log(output_.get(target));
    }

    /**
     * <pre>{@code void Model::computeHidden(const std::vector<int32_t>& input, Vector& hidden) const {
     *  assert(hidden.size() == hsz_);
     *  hidden.zero();
     *  for (auto it = input.cbegin(); it != input.cend(); ++it) {
     *      if(quant_) {
     *          hidden.addRow(*qwi_, *it);
     *      } else {
     *          hidden.addRow(*wi_, *it);
     *      }
     *  }
     *  hidden.mul(1.0 / input.size());
     * }}</pre>
     *
     * @param input
     * @param hidden
     */
    public void computeHidden(final List<Integer> input, Vector hidden) {
        Utils.checkArgument(hidden.size() == hsz_);
        hidden.zero();
        for (Integer it : input) {
            if (quant_) {
                hidden.addRow(qwi_, it);
            } else {
                hidden.addRow(wi_, it);
            }
        }
        hidden.mul(1.0f / input.size());
    }

    /**
     * <pre>{@code
     * void Model::predict(const std::vector<int32_t>& input, int32_t k, std::vector<std::pair<real, int32_t>>& heap, Vector& hidden, Vector& output) const {
     *  assert(k > 0);
     *  heap.reserve(k + 1);
     *  computeHidden(input, hidden);
     *  if (args_->loss == loss_name::hs) {
     *      dfs(k, 2 * osz_ - 2, 0.0, heap, hidden);
     *  } else {
     *      findKBest(k, heap, hidden, output);
     *  }
     *  std::sort_heap(heap.begin(), heap.end(), comparePairs);
     * }}</pre>
     *
     * @param input
     * @param k
     * @param heap
     * @param hidden
     * @param output
     */
    public void predict(List<Integer> input, int k, List<Pair<Float, Integer>> heap, Vector hidden, Vector output) {
        Utils.checkArgument(k > 0);
        if (heap instanceof ArrayList) {
            ((ArrayList) heap).ensureCapacity(k + 1);
        }
        computeHidden(input, hidden);
        if (args_.loss == Args.LossName.HS) {
            dfs(k, 2 * osz_ - 2, 0.0f, heap, hidden);
        } else {
            findKBest(k, heap, hidden, output);
        }
        heap.sort(COMPARE_PAIRS);
    }

    public Map<Float, Integer> _predict(List<Integer> input, int k, Vector hidden, Vector output) {
        NavigableMap<Float, Integer> heap = new TreeMap<>(HEAP_COMPARATOR);
        computeHidden(input, hidden);
        if (args_.loss == Args.LossName.HS) {
            _dfs(k, 2 * osz_ - 2, 0.0f, heap, hidden);
        } else {
            _findKBest(k, heap, hidden, output);
        }
        return heap;
    }

    /**
     * <pre>{@code
     * void Model::predict(const std::vector<int32_t>& input, int32_t k, std::vector<std::pair<real, int32_t>>& heap) {
     *  predict(input, k, heap, hidden_, output_);
     * }}</pre>
     *
     * @param input
     * @param k
     * @return
     */
    public Map<Float, Integer> _predict(final List<Integer> input, int k) {
        return _predict(input, k, hidden_, output_);
    }

    /**
     * <pre>{@code
     * void Model::findKBest(int32_t k, std::vector<std::pair<real, int32_t>>& heap, Vector& hidden, Vector& output) const {
     *  computeOutputSoftmax(hidden, output);
     *  for (int32_t i = 0; i < osz_; i++) {
     *  if (heap.size() == k && log(output[i]) < heap.front().first) {
     *      continue;
     *  }
     *  heap.push_back(std::make_pair(log(output[i]), i));
     *  std::push_heap(heap.begin(), heap.end(), comparePairs);
     *  if (heap.size() > k) {
     *      std::pop_heap(heap.begin(), heap.end(), comparePairs);
     *      heap.pop_back();
     *  }
     * }
     * }}</pre>
     *
     * @param k
     * @param heap
     * @param hidden
     * @param output
     */
    private void _findKBest(int k, NavigableMap<Float, Integer> heap, Vector hidden, Vector output) {
        computeOutputSoftmax(hidden, output);
        for (int i = 0; i < osz_; i++) {
            float key = log(output.get(i));
            if (heap.size() == k && key < heap.firstKey()) {
                continue;
            }
            heap.put(key, i);
            if (heap.size() > k) {
                heap.remove(heap.lastKey());
            }
        }
    }

    /**
     * <pre>{@code
     * void Model::dfs(int32_t k, int32_t node, real score, std::vector<std::pair<real, int32_t>>& heap, Vector& hidden) const {
     *  if (heap.size() == k && score < heap.front().first) {
     *      return;
     *  }
     *  if (tree[node].left == -1 && tree[node].right == -1) {
     *      heap.push_back(std::make_pair(score, node));
     *      std::push_heap(heap.begin(), heap.end(), comparePairs);
     *      if (heap.size() > k) {
     *          std::pop_heap(heap.begin(), heap.end(), comparePairs);
     *          heap.pop_back();
     *      }
     *      return;
     *  }
     *  real f;
     *  if (quant_ && args_->qout) {
     *      f = sigmoid(qwo_->dotRow(hidden, node - osz_));
     *  } else {
     *      f = sigmoid(wo_->dotRow(hidden, node - osz_));
     *  }
     *  dfs(k, tree[node].left, score + log(1.0 - f), heap, hidden);
     *  dfs(k, tree[node].right, score + log(f), heap, hidden);
     * }}</pre>
     *
     * @param k
     * @param node
     * @param score
     * @param heap
     * @param hidden
     */
    private void _dfs(int k, int node, float score, NavigableMap<Float, Integer> heap, Vector hidden) {
        if (heap.size() == k && score < heap.firstKey()) {
            return;
        }
        if (tree.get(node).left == -1 && tree.get(node).right == -1) {
            heap.put(score, node);
            if (heap.size() > k) {
                heap.remove(heap.lastKey());
            }
            return;
        }
        float f;
        if (quant_ && args_.qout) {
            f = sigmoid(qwo_.dotRow(hidden, node - osz_));
        } else {
            f = sigmoid(wo_.dotRow(hidden, node - osz_));
        }
        _dfs(k, tree.get(node).left, score + log(1.0f - f), heap, hidden);
        _dfs(k, tree.get(node).right, score + log(f), heap, hidden);
    }

    public void predict(final List<Integer> input, int k, List<Pair<Float, Integer>> heap) {
        predict(input, k, heap, hidden_, output_);
    }

    /**
     * <pre>{@code
     * void Model::findKBest(int32_t k, std::vector<std::pair<real, int32_t>>& heap, Vector& hidden, Vector& output) const {
     *  computeOutputSoftmax(hidden, output);
     *  for (int32_t i = 0; i < osz_; i++) {
     *      if (heap.size() == k && log(output[i]) < heap.front().first) {
     *          continue;
     *      }
     *      heap.push_back(std::make_pair(log(output[i]), i));
     *      std::push_heap(heap.begin(), heap.end(), comparePairs);
     *      if (heap.size() > k) {
     *          std::pop_heap(heap.begin(), heap.end(), comparePairs);
     *          heap.pop_back();
     *      }
     *  }
     * }}</pre>
     *
     * @param k
     * @param heap
     * @param hidden
     * @param output
     */
    public void findKBest(int k, List<Pair<Float, Integer>> heap, Vector hidden, Vector output) {
        computeOutputSoftmax(hidden, output);
        for (int i = 0; i < osz_; i++) {
            if (heap.size() == k && log(output.get(i)) < heap.get(heap.size() - 1).first()) {
                continue;
            }
            heap.add(new Pair<>(log(output.get(i)), i));

            // TODO: is it correct ? does it make sense?
            heap.sort(COMPARE_PAIRS);
            if (heap.size() > k) {
                heap.sort(COMPARE_PAIRS);
                heap.remove(heap.size() - 1); // pop last
            }
        }
    }

    /**
     * <pre>{@code
     * void Model::dfs(int32_t k, int32_t node, real score, std::vector<std::pair<real, int32_t>>& heap, Vector& hidden) const {
     *  if (heap.size() == k && score < heap.front().first) {
     *      return;
     *  }
     *  if (tree[node].left == -1 && tree[node].right == -1) {
     *      heap.push_back(std::make_pair(score, node));
     *      std::push_heap(heap.begin(), heap.end(), comparePairs);
     *      if (heap.size() > k) {
     *          std::pop_heap(heap.begin(), heap.end(), comparePairs);
     *          heap.pop_back();
     *      }
     *      return;
     *  }
     *  real f;
     *  if (quant_ && args_->qout) {
     *      f = sigmoid(qwo_->dotRow(hidden, node - osz_));
     *  } else {
     *      f = sigmoid(wo_->dotRow(hidden, node - osz_));
     *  }
     *  dfs(k, tree[node].left, score + log(1.0 - f), heap, hidden);
     *  dfs(k, tree[node].right, score + log(f), heap, hidden);
     * }}</pre>
     *
     * @param k
     * @param node
     * @param score
     * @param heap
     * @param hidden
     */
    public void dfs(int k, int node, float score, List<Pair<Float, Integer>> heap, Vector hidden) {
        if (heap.size() == k && score < heap.get(heap.size() - 1).first()) {
            return;
        }
        if (tree.get(node).left == -1 && tree.get(node).right == -1) {
            heap.add(new Pair<>(score, node));
            heap.sort(COMPARE_PAIRS);
            if (heap.size() > k) {
                heap.sort(COMPARE_PAIRS);
                heap.remove(heap.size() - 1); // pop last
            }
            return;
        }
        float f;
        if (quant_ && args_.qout) {
            f = sigmoid(qwo_.dotRow(hidden, node - osz_));
        } else {
            f = sigmoid(wo_.dotRow(hidden, node - osz_));
        }
        dfs(k, tree.get(node).left, score + log(1.0f - f), heap, hidden);
        dfs(k, tree.get(node).right, score + log(f), heap, hidden);
    }

    /**
     * <pre>{@code void Model::update(const std::vector<int32_t>& input, int32_t target, real lr) {
     *  assert(target >= 0);
     *  assert(target < osz_);
     *  if (input.size() == 0) return;
     *  computeHidden(input, hidden_);
     *  if (args_->loss == loss_name::ns) {
     *      loss_ += negativeSampling(target, lr);
     *  } else if (args_->loss == loss_name::hs) {
     *      loss_ += hierarchicalSoftmax(target, lr);
     *  } else {
     *      loss_ += softmax(target, lr);
     *  }
     *  nexamples_ += 1;
     *  if (args_->model == model_name::sup) {
     *      grad_.mul(1.0 / input.size());
     *  }
     *  for (auto it = input.cbegin(); it != input.cend(); ++it) {
     *      wi_->addRow(grad_, *it, 1.0);
     *  }
     * }}</pre>
     *
     * @param input
     * @param target
     * @param lr
     */
    public void update(final List<Integer> input, int target, float lr) {
        Utils.checkArgument(target >= 0);
        Utils.checkArgument(target < osz_);
        if (input.size() == 0) {
            return;
        }
        computeHidden(input, hidden_);
        if (args_.loss == LossName.NS) {
            loss_ += negativeSampling(target, lr);
        } else if (args_.loss == Args.LossName.HS) {
            loss_ += hierarchicalSoftmax(target, lr);
        } else {
            loss_ += softmax(target, lr);
        }
        nexamples_ += 1;
        if (args_.model == ModelName.SUP) {
            grad_.mul(1.0f / input.size());
        }
        for (Integer it : input) {
            wi_.addRow(grad_, it, 1.0f);
        }
    }

    /**
     * <pre>{@code
     * void Model::setTargetCounts(const std::vector<int64_t>& counts) {
     *  assert(counts.size() == osz_);
     *  if (args_->loss == loss_name::ns) {
     *      initTableNegatives(counts);
     *  }
     *  if (args_->loss == loss_name::hs) {
     *      buildTree(counts);
     *  }
     * }
     * }</pre>
     *
     * @param counts
     */
    public void setTargetCounts(final List<Long> counts) {
        Utils.checkArgument(counts.size() == osz_);
        if (args_.loss == Args.LossName.NS) {
            initTableNegatives(counts);
        }
        if (args_.loss == Args.LossName.HS) {
            buildTree(counts);
        }
    }

    /**
     * <pre>{@code
     * void Model::initTableNegatives(const std::vector<int64_t>& counts) {
     *  real z = 0.0;
     *  for (size_t i = 0; i < counts.size(); i++) {
     *      z += pow(counts[i], 0.5);
     *  }
     *  for (size_t i = 0; i < counts.size(); i++) {
     *      real c = pow(counts[i], 0.5);
     *      for (size_t j = 0; j < c * NEGATIVE_TABLE_SIZE / z; j++) {
     *          negatives.push_back(i);
     *      }
     *  }
     *  std::shuffle(negatives.begin(), negatives.end(), rng);
     * }
     * }</pre>
     *
     * @param counts
     */
    public void initTableNegatives(final List<Long> counts) {
        if (negatives == null) {
            negatives = new ArrayList<>(counts.size());
        }
        float z = 0.0f;
        for (Long count : counts) {
            z += (float) Math.pow(count, 0.5f);
        }
        for (int i = 0; i < counts.size(); i++) {
            float c = (float) Math.pow(counts.get(i), 0.5f);
            for (int j = 0; j < c * NEGATIVE_TABLE_SIZE / z; j++) {
                negatives.add(i);
            }
        }
        Collections.shuffle(negatives, rng);
    }

    public int getNegative(int target) {
        int negative;
        do {
            negative = negatives.get(negpos);
            negpos = (negpos + 1) % negatives.size();
        } while (target == negative);
        return negative;
    }

    public void buildTree(final List<Long> counts) {
        paths = new ArrayList<>(osz_);
        codes = new ArrayList<>(osz_);
        tree = new ArrayList<>(2 * osz_ - 1);

        for (int i = 0; i < 2 * osz_ - 1; i++) {
            Node node = new Node();
            node.parent = -1;
            node.left = -1;
            node.right = -1;
            node.count = 1000000000000000L;// 1e15f;
            node.binary = false;
            tree.add(i, node);
        }
        for (int i = 0; i < osz_; i++) {
            tree.get(i).count = counts.get(i);
        }
        int leaf = osz_ - 1;
        int node = osz_;
        for (int i = osz_; i < 2 * osz_ - 1; i++) {
            int[] mini = new int[2];
            for (int j = 0; j < 2; j++) {
                if (leaf >= 0 && tree.get(leaf).count < tree.get(node).count) {
                    mini[j] = leaf--;
                } else {
                    mini[j] = node++;
                }
            }
            tree.get(i).left = mini[0];
            tree.get(i).right = mini[1];
            tree.get(i).count = tree.get(mini[0]).count + tree.get(mini[1]).count;
            tree.get(mini[0]).parent = i;
            tree.get(mini[1]).parent = i;
            tree.get(mini[1]).binary = true;
        }
        for (int i = 0; i < osz_; i++) {
            List<Integer> path = new ArrayList<Integer>();
            List<Boolean> code = new ArrayList<Boolean>();
            int j = i;
            while (tree.get(j).parent != -1) {
                path.add(tree.get(j).parent - osz_);
                code.add(tree.get(j).binary);
                j = tree.get(j).parent;
            }
            paths.add(path);
            codes.add(code);
        }
    }

    public float getLoss() {
        return loss_ / nexamples_;
    }

    /**
     * <pre>{@code void Model::initSigmoid() {
     *  t_sigmoid = new real[SIGMOID_TABLE_SIZE + 1];
     *  for (int i = 0; i < SIGMOID_TABLE_SIZE + 1; i++) {
     *      real x = real(i * 2 * MAX_SIGMOID) / SIGMOID_TABLE_SIZE - MAX_SIGMOID;
     *      t_sigmoid[i] = 1.0 / (1.0 + std::exp(-x));
     *  }
     * }}</pre>
     */
    private void initSigmoid() {
        t_sigmoid = new float[SIGMOID_TABLE_SIZE + 1];
        for (int i = 0; i < SIGMOID_TABLE_SIZE + 1; i++) {
            float x = i * 2f * MAX_SIGMOID / SIGMOID_TABLE_SIZE - MAX_SIGMOID;
            t_sigmoid[i] = (float) (1 / (1 + Math.exp(-x)));
        }
    }

    /**
     * <pre>{@code void Model::initLog() {
     *  t_log = new real[LOG_TABLE_SIZE + 1];
     *  for (int i = 0; i < LOG_TABLE_SIZE + 1; i++) {
     *      real x = (real(i) + 1e-5) / LOG_TABLE_SIZE;
     *      t_log[i] = std::log(x);
     *  }
     * }}</pre>
     */
    private void initLog() {
        t_log = new float[LOG_TABLE_SIZE + 1];
        for (int i = 0; i < LOG_TABLE_SIZE + 1; i++) {
            float x = (i + 1e-5f) / LOG_TABLE_SIZE;
            t_log[i] = (float) Math.log(x);
        }
    }

    /**
     * <pre>{@code real Model::log(real x) const {
     *  if (x > 1.0) {
     *      return 0.0;
     *  }
     *  int i = int(x * LOG_TABLE_SIZE);
     *  return t_log[i];
     * }}</pre>
     *
     * @param x
     * @return
     */
    public float log(float x) {
        if (x > 1.0f) {
            return 0.0f;
        }
        int i = (int) (x * LOG_TABLE_SIZE);
        return t_log[i];
    }

    public float sigmoid(float x) {
        if (x < -MAX_SIGMOID) {
            return 0.0f;
        } else if (x > MAX_SIGMOID) {
            return 1.0f;
        } else {
            int i = (int) ((x + MAX_SIGMOID) * SIGMOID_TABLE_SIZE / MAX_SIGMOID / 2);
            return t_sigmoid[i];
        }
    }

    public class Node {
        int parent;
        int left;
        int right;
        long count;
        boolean binary;
    }
}
