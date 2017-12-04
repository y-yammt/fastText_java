package cc.fasttext;

import java.io.IOException;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Objects;
import java.util.function.DoubleUnaryOperator;
import java.util.stream.Collectors;

import org.apache.commons.lang.Validate;
import org.apache.commons.math3.distribution.UniformRealDistribution;
import org.apache.commons.math3.random.RandomGenerator;
import org.apache.commons.math3.util.FastMath;

import ru.avicomp.io.FTInputStream;
import ru.avicomp.io.FTOutputStream;

/**
 * see <a href='https://github.com/facebookresearch/fastText/blob/master/src/model.cc'>matrix.cc</a> and
 * <a href='https://github.com/facebookresearch/fastText/blob/master/src/model.h'>matrix.h</a>
 */
public strictfp class Matrix {

    private float[][] data_;
    // todo: make final ?
    protected int m_; // vocabSize
    protected int n_; // layer1Size

    Matrix() {
        // empty
    }

    public Matrix(int m, int n) {
        Validate.isTrue(m > 0, "Wrong m-size: " + m);
        Validate.isTrue(n > 0, "Wrong n-size: " + n);
        m_ = m;
        n_ = n;
        data_ = new float[m][n];
    }

    public Matrix copy() {
        Matrix res = new Matrix(m_, n_);
        for (int i = 0; i < m_; i++) {
            System.arraycopy(data_[i], 0, res.data_[i], 0, n_);
        }
        return res;
    }

    float[] flatData() {
        float[] res = new float[m_ * n_];
        for (int i = 0; i < m_; i++) {
            System.arraycopy(data_[i], 0, res, i * n_, n_);
        }
        return res;
    }

    float[][] data() {
        return data_;
    }

    public List<Vector> getData() {
        return Collections.unmodifiableList(Arrays.stream(data_).map(Vector::new).collect(Collectors.toList()));
    }

    public boolean isEmpty() {
        return m_ == 0 || n_ == 0;
    }

    public int getM() {
        return m_;
    }

    public int getN() {
        return n_;
    }

    public float get(int i, int j) {
        validateMIndex(i);
        validateNIndex(j);
        return at(i, j);
    }

    float at(int i, int j) {
        return data_[i][j];
    }

    public void set(int i, int j, float value) {
        validateMIndex(i);
        validateNIndex(j);
        put(i, j, value);
    }

    void put(int i, int j, float value) {
        data_[i][j] = value;
    }

    public void compute(int i, int j, DoubleUnaryOperator operator) {
        Objects.requireNonNull(operator, "Null operator");
        data_[i][j] = (float) operator.applyAsDouble(data_[i][j]);
    }

    void validateMIndex(int i) {
        Validate.isTrue(i >= 0 && i < m_, "First index (" + i + ") is out of range [0, " + m_ + ")");
    }

    void validateNIndex(int j) {
        Validate.isTrue(j >= 0 && j < n_, "Second index (" + j + ") is out of range [0, " + n_ + ")");
    }

    void validateNVector(Vector vector) {
        Validate.isTrue(Objects.requireNonNull(vector, "Null vector").size() == n_, "Wrong vector size: " + vector.size() + " (!= " + n_ + ")");
    }

    void validateMVector(Vector vector) {
        Validate.isTrue(Objects.requireNonNull(vector, "Null vector").size() == m_, "Wrong vector size: " + vector.size() + " (!= " + m_ + ")");
    }

    /**
     * <pre>{@code
     * void Matrix::uniform(real a) {
     *  std::minstd_rand rng(1);
     *  std::uniform_real_distribution<> uniform(-a, a);
     *  for (int64_t i = 0; i < (m_ * n_); i++) {
     *      data_[i] = uniform(rng);
     *  }
     * }
     * }</pre>
     *
     * @param rnd
     * @param a
     */
    public void uniform(RandomGenerator rnd, float a) {
        UniformRealDistribution uniform = new UniformRealDistribution(rnd, -a, a);
        for (int i = 0; i < m_; i++) {
            for (int j = 0; j < n_; j++) {
                data_[i][j] = (float) uniform.sample();
            }
        }
    }

    /**
     * <pre>{@code real Matrix::dotRow(const Vector& vec, int64_t i) const {
     *  assert(i >= 0);
     *  assert(i < m_);
     *  assert(vec.size() == n_);
     *  real d = 0.0;
     *  for (int64_t j = 0; j < n_; j++) {
     *      d += at(i, j) * vec.data_[j];
     *  }
     *  return d;
     * }}</pre>
     *
     * @param vector
     * @param i
     * @return
     */
    public float dotRow(Vector vector, int i) {
        validateMIndex(i);
        validateNVector(vector);
        float d = 0f;
        for (int j = 0; j < getN(); j++) {
            d += data_[i][j] * vector.get(j);
        }
        return d;
    }

    /**
     * <pre>{@code void Matrix::addRow(const Vector& vec, int64_t i, real a) {
     *  assert(i >= 0);
     *  assert(i < m_);
     *  assert(vec.size() == n_);
     *  for (int64_t j = 0; j < n_; j++) {
     *      data_[i * n_ + j] += a * vec.data_[j];
     *  }
     * }}</pre>
     *
     * @param vector
     * @param i
     * @param a
     */
    public void addRow(Vector vector, int i, float a) {
        validateMIndex(i);
        validateNVector(vector);
        for (int j = 0; j < getN(); j++) {
            data_[i][j] += a * vector.get(j);
        }
    }

    /**
     * <pre>{@code void Matrix::multiplyRow(const Vector& nums, int64_t ib, int64_t ie) {
     *  if (ie == -1) {
     *      ie = m_;
     *  }
     *  assert(ie <= nums.size());
     *  for (auto i = ib; i < ie; i++) {
     *      real n = nums[i-ib];
     *      if (n != 0) {
     *          for (auto j = 0; j < n_; j++) {
     *              at(i, j) *= n;
     *          }
     *      }
     *  }
     * }}</pre>
     *
     * @param denoms
     * @param ib
     * @param ie
     */
    protected void multiplyRow(Vector denoms, int ib, int ie) {
        if (ie == -1) {
            ie = m_;
        }
        Validate.isTrue(ie <= denoms.size());
        for (int i = ib; i < ie; i++) {
            float n = denoms.get(i - ib);
            if (n == 0) {
                continue;
            }
            DoubleUnaryOperator op = v -> v * n;
            for (int j = 0; j < n_; j++) {
                compute(i, j, op);
            }
        }
    }

    public void multiplyRow(Vector denoms) {
        multiplyRow(denoms, 0, -1);
    }

    /**
     * <pre>{@code void Matrix::divideRow(const Vector& denoms, int64_t ib, int64_t ie) {
     *  if (ie == -1) {
     *      ie = m_;
     *  }
     *  assert(ie <= denoms.size());
     *  for (auto i = ib; i < ie; i++) {
     *      real n = denoms[i-ib];
     *      if (n != 0) {
     *          for (auto j = 0; j < n_; j++) {
     *              at(i, j) /= n;
     *          }
     *      }
     *  }
     * }}</pre>
     *
     * @param denoms
     * @param ib
     * @param ie
     */
    protected void divideRow(Vector denoms, int ib, int ie) {
        if (ie == -1) {
            ie = m_;
        }
        Validate.isTrue(ie <= denoms.size());
        for (int i = ib; i < ie; i++) {
            float n = denoms.get(i - ib);
            if (n == 0) {
                continue;
            }
            DoubleUnaryOperator op = v -> v / n;
            for (int j = 0; j < n_; j++) {
                compute(i, j, op);
            }
        }
    }

    public void divideRow(Vector denoms) {
        divideRow(denoms, 0, -1);
    }

    /**
     * <pre>{@code real Matrix::l2NormRow(int64_t i) const {
     *  auto norm = 0.0;
     *  for (auto j = 0; j < n_; j++) {
     *      const real v = at(i,j);
     *      norm += v * v;
     *  }
     *  return std::sqrt(norm);
     * }}</pre>
     *
     * @param i
     * @return
     */
    private float l2NormRow(int i) {
        double norm = 0.0;
        for (int j = 0; j < n_; j++) {
            float v = at(i, j);
            norm += v * v;
        }
        return (float) FastMath.sqrt(norm);
    }

    /**
     * <pre>{@code void Matrix::l2NormRow(Vector& norms) const {
     *  assert(norms.size() == m_);
     *  for (auto i = 0; i < m_; i++) {
     *      norms[i] = l2NormRow(i);
     *  }
     * }}</pre>
     *
     * @return
     */
    public Vector l2NormRow() {
        Vector res = new Vector(m_);
        for (int i = 0; i < m_; i++) {
            res.set(i, l2NormRow(i));
        }
        return res;
    }

    /**
     * <pre>{@code void Matrix::l2NormRow(Vector& norms) const {
     *  assert(norms.size() == m_);
     *  for (auto i = 0; i < m_; i++) {
     *      norms[i] = l2NormRow(i);
     *  }
     * }}</pre>
     *
     * @param norms
     */
    protected void l2NormRow(Vector norms) {
        validateMVector(norms);
        for (int i = 0; i < m_; i++) {
            norms.set(i, l2NormRow(i));
        }
    }


    /**
     * <pre>{@code
     * void Matrix::save(std::ostream& out) {
     *  out.write((char*) &m_, sizeof(int64_t));
     *  out.write((char*) &n_, sizeof(int64_t));
     *  out.write((char*) data_, m_ * n_ * sizeof(real));
     * }}</pre>
     *
     * @param out {@link FTOutputStream}
     * @throws IOException if an I/O error occurs
     */
    void save(FTOutputStream out) throws IOException {
        out.writeLong(m_);
        out.writeLong(n_);
        for (int i = 0; i < m_; i++) {
            for (int j = 0; j < n_; j++) {
                out.writeFloat(data_[i][j]);
            }
        }
    }

    /**
     * <pre>{@code void Matrix::load(std::istream& in) {
     *  in.read((char*) &m_, sizeof(int64_t));
     *  in.read((char*) &n_, sizeof(int64_t));
     *  delete[] data_;
     *  data_ = new real[m_ * n_];
     *  in.read((char*) data_, m_ * n_ * sizeof(real));
     * }}</pre>
     *
     * @param in {@link FTInputStream}
     * @throws IOException if an I/O error occurs
     */
    void load(FTInputStream in) throws IOException {
        m_ = (int) in.readLong();
        n_ = (int) in.readLong();
        data_ = new float[m_][n_];
        for (int i = 0; i < m_; i++) {
            for (int j = 0; j < n_; j++) {
                data_[i][j] = in.readFloat();
            }
        }
    }

    @Override
    public String toString() {
        return String.format("%s[(n)%dx(m)%d]%s", getClass().getSimpleName(), n_, m_, getData());
    }
}
