package cc.fasttext;

import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.function.IntFunction;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

import org.apache.commons.math3.distribution.UniformRealDistribution;
import org.apache.commons.math3.random.RandomAdaptor;
import org.apache.commons.math3.random.RandomGenerator;
import org.apache.commons.math3.util.FastMath;

import com.google.common.primitives.Bytes;
import com.google.common.primitives.Floats;
import com.google.common.primitives.Ints;

/**
 * TODO: implement
 * see <a href='https://github.com/facebookresearch/fastText/blob/master/src/productquantizer.cc'>productquantizer.cc</a> and
 * <a href='https://github.com/facebookresearch/fastText/blob/master/src/productquantizer.h'>productquantizer.h</>
 * Created by @szuev on 27.10.2017.
 */
public class ProductQuantizer {

    private final int nbits_ = 8;
    private final int ksub_ = 1 << nbits_;
    private final int max_points_per_cluster_ = 256;
    private final int max_points_ = max_points_per_cluster_ * ksub_;
    private final int seed_ = 1234;
    private final int niter_ = 25;
    final double eps_ = 1e-7d;

    private int dim_;
    private int nsubq_;
    private int dsub_;
    private int lastdsub_;

    private float[][] centroids_;

    private RandomGenerator rng;

    /**
     * <pre>{@code ProductQuantizer::ProductQuantizer(int32_t dim, int32_t dsub):
     *  dim_(dim), nsubq_(dim / dsub), dsub_(dsub), centroids_(dim * ksub_), rng(seed_) {
     *  lastdsub_ = dim_ % dsub;
     *  if (lastdsub_ == 0) {lastdsub_ = dsub_;}
     *  else {nsubq_++;}
     * }
     * }</pre>
     *
     * @param randomProvider
     * @param dim
     * @param dsub
     */
    public ProductQuantizer(IntFunction<RandomGenerator> randomProvider, int dim, int dsub) {
        this.dim_ = dim;
        this.nsubq_ = dim / dsub;
        this.dsub_ = dsub;
        this.centroids_ = new float[dim * ksub_][];
        this.rng = randomProvider.apply(seed_);
        this.lastdsub_ = dim_ % dsub;
        if (this.lastdsub_ == 0) {
            this.lastdsub_ = dsub_;
        } else {
            this.nsubq_++;
        }
    }

    /**
     * <pre>{@code real* ProductQuantizer::get_centroids(int32_t m, uint8_t i) {
     *  if (m == nsubq_ - 1) {
     *      return &centroids_[m * ksub_ * dsub_ + i * lastdsub_];
     *  }
     *  return &centroids_[(m * ksub_ + i) * dsub_];
     * }}</pre>
     *
     * @param m
     * @param i
     * @return
     */
    public float[] getCentroids(int m, byte i) {
        int index = m == nsubq_ - 1 ? m * ksub_ * dsub_ + i * lastdsub_ : (m * ksub_ + i) * dsub_;
        return centroids_[index];
    }

    public List<Float> getCentroidsAsList(int m, byte i) {
        return asFloatList(getCentroids(m, i));
    }


    /**
     * <pre>{@code real distL2(const real* x, const real* y, int32_t d) {
     *  real dist = 0;
     *  for (auto i = 0; i < d; i++) {
     *      auto tmp = x[i] - y[i];
     *      dist += tmp * tmp;
     *  }
     *  return dist;
     * }}</pre>
     *
     * @param x
     * @param y
     * @param d
     * @return
     */
    private float distL2(List<Float> x, List<Float> y, int d) {
        float dist = 0;
        for (int i = 0; i < d; i++) {
            float tmp = getFloat(x, i) - getFloat(y, i);
            dist += tmp * tmp;
        }
        return dist;
    }

    /**
     * <pre>{@code
     * real ProductQuantizer::assign_centroid(const real * x, const real* c0, uint8_t* code, int32_t d) const {
     *  const real* c = c0;
     *  real dis = distL2(x, c, d);
     *  code[0] = 0;
     *  for (auto j = 1; j < ksub_; j++) {
     *      c += d;
     *      real disij = distL2(x, c, d);
     *      if (disij < dis) {
     *          code[0] = (uint8_t) j;
     *          dis = disij;
     *      }
     *  }
     *  return dis;
     * }}</pre>
     *
     * @param x
     * @param c0
     * @param code
     * @param d
     * @return
     */
    private float assignCentroid(List<Float> x, float[] c0, List<Byte> code, int d) {
        List<Float> c = asFloatList(c0);
        float dis = distL2(x, c, d);
        code.set(0, (byte) 0);
        for (int j = 1; j < ksub_; j++) {
            c = shift(c, d);
            float disij = distL2(x, c, d);
            if (disij < dis) {
                code.set(0, (byte) j);
                dis = disij;
            }
        }
        return dis;
    }


    /**
     * <pre>{@code
     * void ProductQuantizer::Estep(const real* x, const real* centroids, uint8_t* codes, int32_t d, int32_t n) const {
     *  for (auto i = 0; i < n; i++) {
     *      assign_centroid(x + i * d, centroids, codes + i, d);
     *  }
     * }}</pre>
     *
     * @param x
     * @param centroids
     * @param codes
     * @param d
     * @param n
     */
    private void eStep(float[] x, float[] centroids, List<Byte> codes, int d, int n) {
        List<Float> _x = asFloatList(x);
        for (int i = 0; i < n; i++) {
            assignCentroid(shift(_x, i * d), centroids, shift(codes, i), d);
        }
    }

    /**
     * <pre>{@code
     * void ProductQuantizer::MStep(const real* x0, real* centroids, const uint8_t* codes, int32_t d, int32_t n) {
     *  std::vector<int32_t> nelts(ksub_, 0);
     *  memset(centroids, 0, sizeof(real) * d * ksub_);
     *  const real* x = x0;
     *  for (auto i = 0; i < n; i++) {
     *      auto k = codes[i];
     *      real* c = centroids + k * d;
     *      for (auto j = 0; j < d; j++) {
     *          c[j] += x[j];
     *      }
     *      nelts[k]++;
     *      x += d;
     *  }
     *  real* c = centroids;
     *  for (auto k = 0; k < ksub_; k++) {
     *      real z = (real) nelts[k];
     *      if (z != 0) {
     *          for (auto j = 0; j < d; j++) {
     *              c[j] /= z;
     *          }
     *      }
     *      c += d;
     *  }
     *  std::uniform_real_distribution<> runiform(0,1);
     *  for (auto k = 0; k < ksub_; k++) {
     *      if (nelts[k] == 0) {
     *          int32_t m = 0;
     *          while (runiform(rng) * (n - ksub_) >= nelts[m] - 1) {
     *              m = (m + 1) % ksub_;
     *          }
     *          memcpy(centroids + k * d, centroids + m * d, sizeof(real) * d);
     *          for (auto j = 0; j < d; j++) {
     *              int32_t sign = (j % 2) * 2 - 1;
     *              centroids[k * d + j] += sign * eps_;
     *              centroids[m * d + j] -= sign * eps_;
     *          }
     *          nelts[k] = nelts[m] / 2;
     *          nelts[m] -= nelts[k];
     *      }
     *  }
     * }}</pre>
     *
     * @param x0
     * @param centroids
     * @param codes
     * @param d
     * @param n
     */
    private void mStep(float[] x0, float[] centroids, List<Byte> codes, int d, int n) {
        List<Integer> nelts = asIntList(new int[ksub_]);
        // `memset(centroids, 0, sizeof(real) * d * ksub_);` :
        Arrays.fill(centroids, 0, d * ksub_, 0);
        List<Float> x = asFloatList(x0);
        List<Float> c = asFloatList(centroids);
        for (int i = 0; i < n; i++) {
            byte k = getByte(codes, i);
            c = shift(c, k * d);
            for (int j = 0; j < d; j++) {
                c.set(j, c.get(j) + x.get(j));
            }
            nelts.set(k, nelts.get(k) + 1);
            x = shift(x, d);
        }
        c = asFloatList(centroids);
        for (int k = 0; k < ksub_; k++) {
            float z = (float) nelts.get(k);
            if (z != 0) {
                for (int j = 0; j < d; j++) {
                    c.set(j, c.get(j) / z);
                }
            }
            c = shift(c, d);
        }

        UniformRealDistribution runiform = new UniformRealDistribution(rng, 0, 1);
        for (int k = 0; k < ksub_; k++) {
            if (nelts.get(k) != 0) continue;
            int m = 0;
            while (runiform.sample() * (n - ksub_) >= nelts.get(m) - 1) {
                m = (m + 1) % ksub_;
            }
            // `memcpy(centroids + k * d, centroids + m * d, sizeof(real) * d)` :
            System.arraycopy(centroids, m * d, centroids, k * d, d);

            for (int j = 0; j < d; j++) {
                int sign = (j % 2) * 2 - 1;
                centroids[k * d + j] += sign * eps_;
                centroids[m * d + j] -= sign * eps_;
            }
            nelts.set(k, nelts.get(m) / 2);
            nelts.set(m, nelts.get(m) - nelts.get(k));
        }
    }


    /**
     * <pre>{@code void ProductQuantizer::train(int32_t n, const real * x) {
     *  if (n < ksub_) {
     *      std::cerr<<"Matrix too small for quantization, must have > 256 rows"<<std::endl;
     *      exit(1);
     *  }
     *  std::vector<int32_t> perm(n, 0);
     *  std::iota(perm.begin(), perm.end(), 0);
     *  auto d = dsub_;
     *  auto np = std::min(n, max_points_);
     *  real* xslice = new real[np * dsub_];
     *  for (auto m = 0; m < nsubq_; m++) {
     *      if (m == nsubq_-1) {
     *          d = lastdsub_;
     *      }
     *      if (np != n) {
     *          std::shuffle(perm.begin(), perm.end(), rng);
     *      }
     *      for (auto j = 0; j < np; j++) {
     *          memcpy (xslice + j * d, x + perm[j] * dim_ + m * dsub_, d * sizeof(real));
     *      }
     *      kmeans(xslice, get_centroids(m, 0), np, d);
     *  }
     *  delete [] xslice;
     * }}</pre>
     *
     * @param n
     * @param x
     */
    public void train(int n, float[] x) {
        if (n < ksub_) {
            throw new IllegalArgumentException("Matrix too small for quantization, must have > 256 rows");
        }
        List<Integer> perm = IntStream.iterate(0, operand -> ++operand).limit(n).boxed().collect(Collectors.toList());
        int d = dsub_;
        int np = FastMath.min(n, max_points_);
        float[] xslice = new float[np * dsub_];
        for (int m = 0; m < nsubq_; m++) {
            if (m == nsubq_ - 1) {
                d = lastdsub_;
            }
            if (np != n) {
                Collections.shuffle(perm, new RandomAdaptor(rng));
            }
            for (int j = 0; j < np; j++) {
                // `memcpy (xslice + j * d, x + perm[j] * dim_ + m * dsub_, d * sizeof(real))` :
                System.arraycopy(x, perm.get(j) * dim_ + m * dsub_, xslice, j * d, d);
            }
            kmeans(xslice, getCentroids(m, (byte) 0), np, d);
        }
    }

    /**
     * <pre>{@code void ProductQuantizer::kmeans(const real *x, real* c, int32_t n, int32_t d) {
     *  std::vector<int32_t> perm(n,0);
     *  std::iota(perm.begin(), perm.end(), 0);
     *  std::shuffle(perm.begin(), perm.end(), rng);
     *  for (auto i = 0; i < ksub_; i++) {
     *      memcpy (&c[i * d], x + perm[i] * d, d * sizeof(real));
     *  }
     *  uint8_t* codes = new uint8_t[n];
     *  for (auto i = 0; i < niter_; i++) {
     *      Estep(x, c, codes, d, n);
     *      MStep(x, c, codes, d, n);
     *  }
     *  delete [] codes;
     * }}</pre>
     */
    private void kmeans(float[] x, float[] c, int n, int d) {
        List<Integer> perm = IntStream.iterate(0, operand -> ++operand).limit(n).boxed().collect(Collectors.toList());
        Collections.shuffle(perm, new RandomAdaptor(rng));
        for (int i = 0; i < ksub_; i++) {
            // `memcpy (&c[i * d], x + perm[i] * d, d * sizeof(real))` :
            System.arraycopy(x, perm.get(i) * d, c, i * d, d);
        }
        List<Byte> codes = asByteList(new byte[n]);
        for (int i = 0; i < niter_; i++) {
            eStep(x, c, codes, d, n);
            mStep(x, c, codes, d, n);
        }
    }

    /**
     * <pre>{@code void ProductQuantizer::compute_code(const real* x, uint8_t* code) const {
     *  auto d = dsub_;
     *  for (auto m = 0; m < nsubq_; m++) {
     *      if (m == nsubq_ - 1) {
     *          d = lastdsub_;
     *      }
     *      assign_centroid(x + m * dsub_, get_centroids(m, 0), code + m, d);
     *  }
     * }}</pre>
     *
     * @param x
     * @param code
     */
    private void computeCode(List<Float> x, List<Byte> code) {
        int d = dsub_;
        for (int m = 0; m < nsubq_; m++) {
            if (m == nsubq_ - 1) {
                d = lastdsub_;
            }
            assignCentroid(shift(x, m * dsub_), getCentroids(m, (byte) 0), shift(code, m), d);
        }
    }

    /**
     * <pre>{@code void ProductQuantizer::compute_codes(const real* x, uint8_t* codes, int32_t n) const {
     *  for (auto i = 0; i < n; i++) {
     *      compute_code(x + i * dim_, codes + i * nsubq_);
     *  }
     * }}</pre>
     *
     * @param x
     * @param code
     * @param n
     */
    public void computeCode(float[] x, byte[] code, int n) {
        List<Float> _x = asFloatList(x);
        List<Byte> _c = asByteList(code);
        for (int i = 0; i < n; i++) {
            computeCode(shift(_x, i * dim_), shift(_c, i * nsubq_));
        }
    }

    public static List<Byte> asByteList(byte... unsignedByteInts) { // uint8_t
        return Bytes.asList(unsignedByteInts);
        //return IntStream.of(unsignedBytes).mapToObj(i -> (byte) i).collect(Collectors.toList());
    }

    public static List<Float> asFloatList(float... values) {
        return Floats.asList(values);
    }

    public static List<Integer> asIntList(int... values) {
        return Ints.asList(values);
    }

    private static float getFloat(List<Float> array, int index) {
        if (index >= array.size()) {
            return Float.NaN;
        }
        return array.get(index);
    }

    private static byte getByte(List<Byte> array, int index) {
        if (index >= array.size()) {
            return Byte.MIN_VALUE;
        }
        return array.get(index);
    }

    public static <T> List<T> shift(List<T> array, int index) {
        if (index >= array.size()) {
            return Collections.emptyList();
        }
        return array.subList(index, array.size());
    }
}
