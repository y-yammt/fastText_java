package cc.fasttext;

import java.util.List;
import java.util.Objects;
import java.util.StringJoiner;

import org.apache.commons.lang.Validate;
import org.apache.commons.math3.util.FastMath;

import com.google.common.primitives.Floats;

public strictfp class Vector {

    public int m_;
    public float[] data_;

    public Vector(int size) {
        m_ = size;
        data_ = new float[size];
    }

    public int size() {
        return m_;
    }

    public void zero() {
        data_ = new float[m_];
        //for (int i = 0; i < m_; i++) data_[i] = 0;
    }

    /**
     * <pre>{@code real Vector::norm() const {
     *  real sum = 0;
     *  for (int64_t i = 0; i < m_; i++) {
     *      sum += data_[i] * data_[i];
     *  }
     *  return std::sqrt(sum);
     * }
     * }</pre>
     *
     * @return float
     */
    public float norm() {
        double sum = 0;
        for (int i = 0; i < m_; i++) {
            sum += data_[i] * data_[i];
        }
        return (float) FastMath.sqrt(sum);
    }

    /**
     * <pre>{@code void Vector::addVector(const Vector& source) {
     *  assert(m_ == source.m_);
     *  for (int64_t i = 0; i < m_; i++) {
     *      data_[i] += source.data_[i];
     *  }
     * }}</pre>
     *
     * @param source {@link Vector}
     */
    public void addVector(Vector source) {
        Validate.isTrue(m_ == Objects.requireNonNull(source, "Null source vector").m_, "Wrong size of vector: " + m_ + "!=" + source.m_);
        for (int i = 0; i < m_; i++) {
            data_[i] += source.data_[i];
        }
    }

    /**
     * <pre>{@code void Vector::addVector(const Vector& source, real s) {
     *  assert(m_ == source.m_);
     *  for (int64_t i = 0; i < m_; i++) {
     *      data_[i] += s * source.data_[i];
     *  }
     * }}</pre>
     *
     * @param source
     * @param s
     */
    public void addVector(Vector source, float s) {
        Validate.isTrue(m_ == Objects.requireNonNull(source, "Null source vector").m_, "Wrong size of vector: " + m_ + "!=" + source.m_);
        for (int i = 0; i < m_; i++) {
            data_[i] += s * source.data_[i];
        }
    }

    /**
     * <pre>{@code
     * void Vector::mul(real a) {
     *  for (int64_t i = 0; i < m_; i++) {
     *      data_[i] *= a;
     *  }
     * }}</pre>
     *
     * @param a
     */
    public void mul(float a) {
        for (int i = 0; i < m_; i++) {
            data_[i] *= a;
        }
    }

    /**
     * <pre>{@code void Vector::addRow(const Matrix& A, int64_t i) {
     *  assert(i >= 0);
     *  assert(i < A.m_);
     *  assert(m_ == A.n_);
     *  for (int64_t j = 0; j < A.n_; j++) {
     *      data_[j] += A.at(i, j);
     *  }
     * }}</pre>
     *
     * @param A
     * @param i
     */
    public void addRow(Matrix A, int i) {
        Utils.checkArgument(i >= 0);
        Utils.checkArgument(i < A.m_);
        Utils.checkArgument(m_ == A.n_);
        for (int j = 0; j < A.n_; j++) {
            data_[j] += A.data_[i][j];
        }
    }

    public void addRow(final Matrix A, int i, float a) {
        Utils.checkArgument(i >= 0);
        Utils.checkArgument(i < A.m_);
        Utils.checkArgument(m_ == A.n_);
        for (int j = 0; j < A.n_; j++) {
            data_[j] += a * A.data_[i][j];
        }
    }

    public void mul(final Matrix A, final Vector vec) {
        Utils.checkArgument(A.m_ == m_);
        Utils.checkArgument(A.n_ == vec.m_);
        for (int i = 0; i < m_; i++) {
            data_[i] = 0.0f;
            for (int j = 0; j < A.n_; j++) {
                data_[i] += A.data_[i][j] * vec.data_[j];
            }
        }
    }

    public int argmax() {
        float max = data_[0];
        int argmax = 0;
        for (int i = 1; i < m_; i++) {
            if (data_[i] > max) {
                max = data_[i];
                argmax = i;
            }
        }
        return argmax;
    }

    public float get(int i) {
        return data_[i];
    }

    public void set(int i, float value) {
        data_[i] = value;
    }

    public List<Float> getData() {
        return Floats.asList(data_);
    }

    /**
     * todo: fix.
     * see {@link Utils#formatNumber(float)}
     *
     * @return
     */
    @Override
    public String toString() {
        StringJoiner res = new StringJoiner(" ");
        for (float data : data_) {
            res.add(Utils.formatNumber(data));
        }
        return res.toString();
    }

}
